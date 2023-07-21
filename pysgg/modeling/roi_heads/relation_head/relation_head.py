# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from pysgg.modeling.roi_heads.relation_head.rel_proposal_network.models import (
    gt_rel_proposal_matching,
    RelationProposalModel,
    filter_rel_pairs,
)
from pysgg.utils.visualize_graph import *
from .inference import make_roi_relation_post_processor

from .roi_relation_feature_extractors import make_roi_relation_feature_extractor

from .roi_relation_predictors import make_roi_relation_predictor
from .sampling import make_roi_relation_samp_processor
from ..box_head.roi_box_feature_extractors import (
    make_roi_box_feature_extractor,
    ResNet50Conv5ROIFeatureExtractor,
)
from pysgg.modeling.roi_heads.relation_head.model_kern import (
    to_onehot,
)

from pysgg.layers.balanced_norm import BalancedNorm1d, LearnableBalancedNorm1d

class ROIRelationHead(torch.nn.Module):
    """
    Generic Relation Head class.
    """

    def __init__(self, cfg, in_channels):
        super(ROIRelationHead, self).__init__()
        self.cfg = cfg.clone()
        if cfg.GLOBAL_SETTING.DATASET_CHOICE == 'VG':
            self.num_obj_cls = cfg.MODEL.ROI_BOX_HEAD.VG_NUM_CLASSES
            self.num_rel_cls = cfg.MODEL.ROI_RELATION_HEAD.VG_NUM_CLASSES
        elif cfg.GLOBAL_SETTING.DATASET_CHOICE == 'GQA':
            self.num_obj_cls = cfg.MODEL.ROI_BOX_HEAD.GQA_200_NUM_CLASSES
            self.num_rel_cls = cfg.MODEL.ROI_RELATION_HEAD.GQA_200_NUM_CLASSES

        # mode
        if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = "predcls"
            else:
                self.mode = "sgcls"
        else:
            self.mode = "sgdet"

        # the fix features head for extracting the instances ROI features for
        # obj detection
        if self.cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR in ["VETOPredictor", "VETOPredictor_MEET"]:
            self.box_feature_extractor = make_roi_box_feature_extractor(cfg, in_channels, for_relation=True)
            feat_dim = 512
        else:
            self.box_feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
            self.union_feature_extractor = make_roi_relation_feature_extractor(cfg, in_channels)
            feat_dim = self.box_feature_extractor.out_channels

        if cfg.MODEL.BALANCED_NORM:
            self.balanced_norm = BalancedNorm1d(51, normalized_probs=False,
                                                with_gradient=False)

        self.predictor = make_roi_relation_predictor(cfg, feat_dim)
        self.post_processor = make_roi_relation_post_processor(cfg)
        #self.loss_evaluator = make_roi_relation_loss_evaluator(cfg)
        self.samp_processor = make_roi_relation_samp_processor(cfg)

        self.rel_prop_on = self.cfg.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.SET_ON
        self.rel_prop_type = self.cfg.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.METHOD

        self.object_cls_refine = cfg.MODEL.ROI_RELATION_HEAD.OBJECT_CLASSIFICATION_REFINE
        self.pass_obj_recls_loss = cfg.MODEL.ROI_RELATION_HEAD.REL_OBJ_MULTI_TASK_LOSS

        # parameters
        self.use_union_box = self.cfg.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION

        self.rel_pn = None
        self.use_relness_ranking = False
        self.use_same_label_with_clser = False
        if self.rel_prop_on:
            if self.rel_prop_type == "rel_pn":
                self.rel_pn = RelationProposalModel(cfg)
                self.use_relness_ranking = (
                    cfg.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.USE_RELATEDNESS_FOR_PREDICTION_RANKING
                )
            if self.rel_prop_type == "pre_clser":
                self.use_same_label_with_clser == cfg.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.USE_SAME_LABEL_WITH_CLSER

    def forward(self, features, proposals, depth_features=None, targets=None, logger=None, x=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes. Note: it has been post-processed (regression, nms) in sgdet mode
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        if self.mode == "predcls":
            # overload the pred logits by the gt label
            device = features[0].device
            for proposal in proposals:
                obj_labels = proposal.get_field("labels")
                proposal.add_field("predict_logits", to_onehot(obj_labels, self.num_obj_cls))
                proposal.add_field("pred_scores", torch.ones(len(obj_labels)).to(device))
                proposal.add_field("pred_labels", obj_labels.to(device))
        if self.training:
            # relation subsamples and assign ground truth label during training
            with torch.no_grad():
                if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
                    (
                        proposals,
                        rel_labels,
                        rel_pair_idxs,
                        gt_rel_binarys_matrix,
                    ) = self.samp_processor.gtbox_relsample(proposals, targets)

                    rel_labels_all = rel_labels
                else:
                    (
                        proposals,
                        rel_labels,
                        rel_labels_all,
                        rel_pair_idxs,
                        gt_rel_binarys_matrix,
                    ) = self.samp_processor.detect_relsample(proposals, targets)
        else:
            rel_labels, rel_labels_all, gt_rel_binarys_matrix = None, None, None
            rel_pair_idxs = self.samp_processor.prepare_test_pairs(
                features[0].device, proposals
            )


        # use box_head to extract features that will be fed to the later predictor processing
        if self.cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR in ["VETOPredictor", "VETOPredictor_MEET"]:
            roi_features, d_2d, x_1d, d_1d = self.box_feature_extractor(features, proposals, depth_features=depth_features)

        else:
            roi_features, d_2d = self.box_feature_extractor(features, proposals)

        rel_pn_loss = None
        relness_matrix = None
        if self.rel_prop_on:
            fg_pair_matrixs = None
            gt_rel_binarys_matrix = None

            if targets is not None:
                fg_pair_matrixs, gt_rel_binarys_matrix = gt_rel_proposal_matching(
                    proposals,
                    targets,
                    self.cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
                    self.cfg.TEST.RELATION.REQUIRE_OVERLAP,
                )
                gt_rel_binarys_matrix = [each.float().cuda() for each in gt_rel_binarys_matrix]


            if self.rel_prop_type == "rel_pn":
                relness_matrix, rel_pn_loss = self.rel_pn(
                    proposals,
                    roi_features,
                    rel_pair_idxs,
                    rel_labels,
                    fg_pair_matrixs,
                    gt_rel_binarys_matrix,
                )

                rel_pair_idxs, rel_labels = filter_rel_pairs(
                    relness_matrix, rel_pair_idxs, rel_labels
                )
                for enti_prop, rel_mat in zip(proposals, relness_matrix):
                    enti_prop.add_field('relness_mat', rel_mat.unsqueeze(-1)) 

        if self.cfg.MODEL.ATTRIBUTE_ON:
            att_features = self.att_feature_extractor(features, proposals)
            roi_features = torch.cat((roi_features, att_features), dim=-1)

        if self.cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR not in ("VETOPredictor", "VETOPredictor_MEET"):
            union_features = self.union_feature_extractor(features, proposals, rel_pair_idxs)
        else:
            union_features = None

        # final classifier that converts the features into predictions
        # should corresponding to all the functions and layers after the self.context class
        rel_pn_labels = rel_labels
        if not self.use_same_label_with_clser:
            rel_pn_labels = rel_labels_all
        custom_rel_labels = None #custom_rel_labels
        cur_chosen_matrix = None #cur_chosen_matrix
        incre_idx_list = None
        if self.cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR in ("VETOPredictor", "VETOPredictor_MEET"):
            obj_refine_logits, relation_logits, add_losses, incre_idx_list, cur_chosen_matrix, custom_rel_labels = self.predictor(
                proposals,
                rel_pair_idxs,
                rel_labels,
                logger,
                roi_features=roi_features,
                roi_depth_features=d_2d
            )
        else:
            obj_refine_logits, relation_logits, add_losses, incre_idx_list = self.predictor(
                proposals,
                rel_pair_idxs,
                rel_labels,
                logger=logger,
                roi_features=roi_features,
                roi_depth_features=d_2d, union_features=union_features, rel_binarys=gt_rel_binarys_matrix
            )
        """
        else:
            obj_refine_logits, relation_logits, add_losses = self.predictor(
                proposals,
                rel_pair_idxs,
                rel_pn_labels,
                gt_rel_binarys_matrix,
                roi_features,
                union_features,
                logger,
            )
        """

        if self.cfg.MODEL.BALANCED_NORM:
            relation_probs_norm, labeling_prob, rel_labels_one_hot_count = self.balanced_norm(relation_logits,
                                                                              rel_labels)
        else:
            relation_probs_norm = None
            labeling_prob = None
            rel_labels_one_hot_count = None
        # proposals, rel_pair_idxs, rel_pn_labels,relness_net_input,roi_features,union_features, None
        # for test
        if not self.training:
            # re-NMS on refined object prediction logits
            if not self.object_cls_refine:
                # if don't use object classification refine, we just use the initial logits
                obj_refine_logits = [prop.get_field("predict_logits") for prop in proposals]

            result = self.post_processor(
                (relation_logits, obj_refine_logits), rel_pair_idxs, proposals, incre_idx_list=incre_idx_list, custom_rel_labels=custom_rel_labels, cur_chosen_matrix=cur_chosen_matrix, ensemble=self.cfg.ENSEMBLE_LEARNING.ENABLED
            )

            return roi_features, result, {}

        else:
            return roi_features, proposals, add_losses


def build_roi_relation_head(cfg, in_channels):
    """
    Constructs a new relation head.
    By default, uses ROIRelationHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIRelationHead(cfg, in_channels)
