# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn as nn
from torch.nn import functional as F

from pysgg.layers import Label_Smoothing_Regression
from pysgg.modeling.matcher import Matcher
from pysgg.modeling.utils import cat
from pysgg.config import cfg
import numpy as np
import pickle

class RelationLossComputation(object):
    """
    Computes the loss for relation triplet.
    Also supports FPN
    """

    def __init__(
        self,
        attri_on,
        num_attri_cat,
        max_num_attri,
        attribute_sampling,
        attribute_bgfg_ratio,
        use_label_smoothing,
        predicate_proportion,
    ):
        """
        Arguments:
            bbox_proposal_matcher (Matcher)
            rel_fg_bg_sampler (RelationPositiveNegativeSampler)
        """
        self.num_classes = 51
        self.balanced_norm = False
        self.balanced_loss = True
        self.pcpl = False #True
        self.ldam_loss = False
        self.adaptive_logits = False
        self.attri_on = attri_on
        self.num_attri_cat = num_attri_cat
        self.max_num_attri = max_num_attri
        self.attribute_sampling = attribute_sampling
        self.attribute_bgfg_ratio = attribute_bgfg_ratio
        self.use_label_smoothing = use_label_smoothing
        # self.pred_weight = (1.0 / torch.FloatTensor([0.5,] + predicate_proportion)).cuda()

        file = "/path/user/sg/PySGG/exp/VETO_predcls/sgcls-VETOPredictor/(2022-11-18_00)veto_sgcls_lr0.0012_sgd_ctxt(resampling)/pred_counter.pkl"

        with open(file, "rb") as f:
            frequency = pickle.load(f)
        if self.use_label_smoothing:
            self.criterion_loss = Label_Smoothing_Regression(e=0.01)
        elif self.balanced_norm:
            num_rels = 51
            class_volume = 1000
            rel_counts_path = "/home/user/sg/VETO/pred_counts.pkl"
            with open(rel_counts_path, 'rb') as fin:
                rel_counts = pickle.load(fin)
            """
            cls_num_list = [frequency[i] for i in range(1, len(frequency)+1)]
            beta = 0.9999
            effective_num = 1.0 - np.power(beta, cls_num_list)
            per_cls_weights = (1.0 - beta) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = np.insert(per_cls_weights, 0, 0.3)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
            self.weight = per_cls_weights
            """
            beta = (class_volume - 1.0) / class_volume
            rel_class_weights = (1.0 - beta) / (1 - (beta ** rel_counts))
            rel_class_weights *= float(num_rels) / np.sum(rel_class_weights)
            rel_class_weights = torch.FloatTensor(rel_class_weights).cuda()
            self.loss_relation_balanced_norm = nn.NLLLoss(weight=rel_class_weights)
            self.criterion_loss = nn.CrossEntropyLoss()
        elif self.balanced_loss:
            num_rels = 51
            class_volume = 1000
            rel_counts_path = "/home/user/VETO/pred_counts.pkl"  #"/home/user/sg/VETO/pred_counts.pkl"
            with open(rel_counts_path, 'rb') as fin:
                rel_counts = pickle.load(fin)
            """
            cls_num_list = [frequency[i] for i in range(1, len(frequency)+1)]
            beta = 0.9999
            effective_num = 1.0 - np.power(beta, cls_num_list)
            per_cls_weights = (1.0 - beta) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = np.insert(per_cls_weights, 0, 0.3)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
            self.weight = per_cls_weights
            """
            beta = (class_volume - 1.0) / class_volume
            rel_class_weights = (1.0 - beta) / (1 - (beta ** rel_counts))
            rel_class_weights *= float(num_rels) / np.sum(rel_class_weights)
            rel_class_weights = torch.FloatTensor(rel_class_weights).cuda()
            self.criterion_loss = nn.CrossEntropyLoss()
            self.criterion_loss_rel = nn.CrossEntropyLoss(weight=rel_class_weights)
        elif self.ldam_loss:
            num_rels = 51
            class_volume = 1000
            rel_counts_path = "/home/user/sg/VETO/pred_counts.pkl"
            with open(rel_counts_path, 'rb') as fin:
                rel_counts = pickle.load(fin)
            """
            cls_num_list = [frequency[i] for i in range(1, len(frequency)+1)]
            beta = 0.9999
            effective_num = 1.0 - np.power(beta, cls_num_list)
            per_cls_weights = (1.0 - beta) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = np.insert(per_cls_weights, 0, 0.3)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
            self.weight = per_cls_weights
            """
            beta = (class_volume - 1.0) / class_volume
            rel_class_weights = (1.0 - beta) / (1 - (beta ** rel_counts))
            rel_class_weights *= float(num_rels) / np.sum(rel_class_weights)
            rel_class_weights = torch.FloatTensor(rel_class_weights).cuda()
            self.criterion_loss = nn.CrossEntropyLoss()
            self.criterion_loss_rel = LDAMLoss(rel_counts, weight=rel_class_weights)
        elif self.adaptive_logits:
            num_rels = 51
            class_volume = 1000
            rel_counts_path = "/home/user/sg/VETO/pred_counts.pkl"
            with open(rel_counts_path, 'rb') as fin:
                rel_counts = pickle.load(fin)
            #beta = (class_volume - 1.0) / class_volume
            #rel_class_weights = (1.0 - beta) / (1 - (beta ** rel_counts))
            #rel_class_weights *= float(num_rels) / np.sum(rel_class_weights)
            #rel_class_weights = torch.FloatTensor(rel_class_weights).cuda()
            self.criterion_loss_rel = Adaptive_logit_adjustment(f=torch.FloatTensor(rel_counts))
            self.criterion_loss = nn.CrossEntropyLoss()
        elif self.pcpl:
            self.center_loss_lambda = 0.03  # center_loss_lambda

            # Center Loss version 1: https://github.com/KaiyangZhou/pytorch-center-loss
            # self.center_loss = CenterLoss(num_classes - 1, feat_dim)

            # Center loss version 2: https://github.com/louis-she/center-loss.pytorch
            self.centers = nn.Parameter(torch.Tensor(self.num_classes, 576).normal_(), requires_grad=False).cuda()

            self.corr_order = torch.tensor([(i, j) for i in range(self.num_classes) for j in range(self.num_classes)])

            #self.pcpl_weight = PCPL(self.num_classes)
            self.criterion_loss = nn.CrossEntropyLoss()
        else:
            self.criterion_loss = nn.CrossEntropyLoss()
            self.criterion_loss_rel = nn.CrossEntropyLoss()
            #self.weight = np.ones((n))
        self.BCE_loss = cfg.MODEL.ROI_RELATION_HEAD.USE_BINARY_LOSS


    def __call__(self, proposals, rel_labels, relation_logits, refine_logits):  #, relation_probs_norm=None, labeling_prob=None, rel_labels_one_hot_count=None):
        """
        Computes the loss for relation triplet.
        This requires that the subsample method has been called beforehand.

        Arguments:
            relation_logits (list[Tensor])
            refine_obj_logits (list[Tensor])

        Returns:
            predicate_loss (Tensor)
            finetune_obj_loss (Tensor)
        """
        if self.attri_on:
            if isinstance(refine_logits[0], (list, tuple)):
                refine_obj_logits, refine_att_logits = refine_logits
            else:
                # just use attribute feature, do not actually predict attribute
                self.attri_on = False
                refine_obj_logits = refine_logits
        else:
            refine_obj_logits = refine_logits

        relation_logits = cat(relation_logits, dim=0)
        refine_obj_logits = cat(refine_obj_logits, dim=0)

        fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        rel_labels = cat(rel_labels, dim=0)
        rel_targets = rel_labels.long()
        fg_idxs, bg_idxs = (rel_labels != 0), (rel_labels == 0)  #from recover...
        if len(torch.nonzero(rel_labels != -1)) == 0:
            loss_relation = None
        else:
            loss_relation = self.criterion_loss_rel(relation_logits[rel_labels != -1],
                                                rel_labels[rel_labels != -1].long())

        loss_refine_obj = self.criterion_loss(refine_obj_logits, fg_labels.long())

        # The following code is used to calcaulate sampled attribute loss
        if self.attri_on:
            refine_att_logits = cat(refine_att_logits, dim=0)
            fg_attributes = cat([proposal.get_field("attributes") for proposal in proposals], dim=0)

            attribute_targets, fg_attri_idx = self.generate_attributes_target(fg_attributes)
            if float(fg_attri_idx.sum()) > 0:
                # have at least one bbox got fg attributes
                refine_att_logits = refine_att_logits[fg_attri_idx > 0]
                attribute_targets = attribute_targets[fg_attri_idx > 0]
            else:
                refine_att_logits = refine_att_logits[0].view(1, -1)
                attribute_targets = attribute_targets[0].view(1, -1)

            loss_refine_att = self.attribute_loss(refine_att_logits, attribute_targets, 
                                             fg_bg_sample=self.attribute_sampling, 
                                             bg_fg_ratio=self.attribute_bgfg_ratio)
            return loss_relation, (loss_refine_obj, loss_refine_att)
        else:
            return loss_relation, loss_refine_obj

    def generate_attributes_target(self, attributes):
        """
        from list of attribute indexs to [1,0,1,0,0,1] form
        """
        assert self.max_num_attri == attributes.shape[1]
        device = attributes.device
        num_obj = attributes.shape[0]

        fg_attri_idx = (attributes.sum(-1) > 0).long()
        attribute_targets = torch.zeros((num_obj, self.num_attri_cat), device=device).float()

        for idx in torch.nonzero(fg_attri_idx).squeeze(1).tolist():
            for k in range(self.max_num_attri):
                att_id = int(attributes[idx, k])
                if att_id == 0:
                    break
                else:
                    attribute_targets[idx, att_id] = 1
        return attribute_targets, fg_attri_idx

    def attribute_loss(self, logits, labels, fg_bg_sample=True, bg_fg_ratio=3):
        if fg_bg_sample:
            loss_matrix = F.binary_cross_entropy_with_logits(logits, labels, reduction='none').view(-1)
            fg_loss = loss_matrix[labels.view(-1) > 0]
            bg_loss = loss_matrix[labels.view(-1) <= 0]

            num_fg = fg_loss.shape[0]
            # if there is no fg, add at least one bg
            num_bg = max(int(num_fg * bg_fg_ratio), 1)   
            perm = torch.randperm(bg_loss.shape[0], device=bg_loss.device)[:num_bg]
            bg_loss = bg_loss[perm]

            return torch.cat([fg_loss, bg_loss], dim=0).mean()
        else:
            attri_loss = F.binary_cross_entropy_with_logits(logits, labels)
            attri_loss = attri_loss * self.num_attri_cat / 20.0
            return attri_loss



class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, input, target):
        target = target.view(-1)

        logpt = F.log_softmax(input)
        logpt = logpt.index_select(-1, target).diag()
        logpt = logpt.view(-1)
        pt = logpt.exp()

        logpt = logpt * self.alpha * (target > 0).float() + logpt * (1 - self.alpha) * (target <= 0).float()

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

class PCPL(nn.Module):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        super(PCPL, self).__init__()
        self.center_loss_lambda = 0.03  #center_loss_lambda

        # Center Loss version 1: https://github.com/KaiyangZhou/pytorch-center-loss
        # self.center_loss = CenterLoss(num_classes - 1, feat_dim)

        # Center loss version 2: https://github.com/louis-she/center-loss.pytorch
        self.centers = nn.Parameter(torch.Tensor(self.num_classes, 576).normal_(), requires_grad=False).cuda()

        self.corr_order = torch.tensor([(i, j) for i in range(self.num_classes) for j in range(self.num_classes)])

    def forward(self, relation_logits_raw, rel_labels):
        #if self.pcpl_center_loss:
        #relation_logits_raw, rel_labels = relation_logits
        assert relation_logits_raw is not None
        # compute center loss
        # Way 1
        # loss_center = self.center_loss(relation_logits_raw[fg_idxs].detach().clone(), rel_labels.long()[fg_idxs]) # only compute loss for non-bg classes
        # loss_center = self.center_loss(relation_logits_raw.detach().clone(), rel_labels.long()) # also compute loss for bg class (class 0)
        # Way 2
        rel_features = relation_logits_raw.clone().detach()
        loss_center = compute_center_loss(rel_features, self.centers, rel_labels.long()) * self.center_loss_lambda
        rel_targets = rel_labels.long()

        # (eq. 2) compute e_{kj}
        corr = torch.norm((self.centers[self.corr_order[:, 0]] - self.centers[self.corr_order[:, 1]]), dim=1)
        # (eq. 3 compute u_i)
        global_corr = torch.cat([(torch.sum(corr_class, dim=0) / self.num_classes).reshape(-1) for corr_class in
                                 torch.split(corr, self.num_classes)])
        # (eq. 4 compute correlation factor tao_i as weight)
        eps = 0.09
        max_corr, min_corr = max(global_corr), min(global_corr)
        corr_factor = (global_corr - min_corr + eps) / (max_corr - min_corr)
        weight = corr_factor.detach()
        return weight, loss_center, rel_features

class LDAMLoss(nn.Module):

    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target, weight=self.weight)


class Adaptive_logit_adjustment(nn.Module):

    def __init__(self, f=None, s=30):
        super(Adaptive_logit_adjustment, self).__init__()
        #m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        #m_list = m_list * (max_m / np.max(m_list))
        #m_list = torch.cuda.FloatTensor(m_list)
        #self.m_list = m_list
        assert s > 0
        self.s = s
        self.qf = torch.FloatTensor(1/torch.log((f/torch.min(f)) + 1)).cuda()
        #self.weight = weight

    def forward(self, x, target):
        cos_theta = x.detach().cpu()
        df = torch.FloatTensor((1-cos_theta)/2).cuda()
        self.qf = self.qf #.to(x.device)
        adjust_logit = self.qf * df
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        #index_float = index.type(torch.cuda.FloatTensor)
        #batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        #batch_m = batch_m.view((-1, 1))
        x_m = x - adjust_logit

        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target)

# for center loss
# https://github.com/louis-she/center-loss.pytorch/blob/master/loss.py
def compute_center_loss(features, centers, targets):
    features = features.view(features.size(0), -1)
    target_centers = centers[targets]
    criterion = torch.nn.MSELoss()
    center_loss = criterion(features, target_centers)
    return center_loss

def make_roi_relation_loss_evaluator(cfg):

    loss_evaluator = RelationLossComputation(
        cfg.MODEL.ATTRIBUTE_ON,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.MAX_ATTRIBUTES,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_SAMPLE,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_RATIO,
        cfg.MODEL.ROI_RELATION_HEAD.LABEL_SMOOTHING_LOSS,
        cfg.MODEL.ROI_RELATION_HEAD.REL_PROP,
    )

    return loss_evaluator
