# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn


#from pysgg.config import cfg
from pysgg.structures.bounding_box import BoxList
from .utils_relation import obj_prediction_nms

import ipdb
class PostProcessor(nn.Module):
    """
    From a set of classification scores, box regression and proposals,
    computes the post-processed boxes, and applies NMS to obtain the
    final results
    """

    def __init__(
            self,
            attribute_on,
            use_gt_box=False,
            later_nms_pred_thres=0.3, cfg=None
    ):
        """
        Arguments:

        """
        super(PostProcessor, self).__init__()
        self.cfg = cfg
        self.attribute_on = attribute_on
        self.use_gt_box = use_gt_box
        self.later_nms_pred_thres = later_nms_pred_thres

        
        self.rel_prop_on = cfg.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.SET_ON
        self.rel_prop_type = cfg.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.METHOD

        self.BCE_loss = cfg.MODEL.ROI_RELATION_HEAD.USE_BINARY_LOSS

        self.use_relness_ranking = False
        if self.rel_prop_type == "rel_pn" and self.rel_prop_on:
            self.use_relness_ranking = cfg.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.USE_RELATEDNESS_FOR_PREDICTION_RANKING


    def forward(self, x, rel_pair_idxs, boxes, custom_rel_labels=None, cur_chosen_matrix=None, incre_idx_list=None, ensemble=False):
        """
        re-NMS on refined object classifcations logits
        and ranking the relationship prediction according to the object and relationship
        classification scores

        Arguments:
            x (tuple[tensor, tensor]): x contains the relation logits
                and finetuned object logits from the relation model.
            rel_pair_idxs （list[tensor]): subject and object indice of each relation,
                the size of tensor is (num_rel, 2)
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra fields labels and scores
        """
        relation_logits, refine_logits = x
        consensus = False
        unanimous = False
        ensemble = False
        if self.cfg.ENSEMBLE_LEARNING.ENABLED:
            if self.cfg.ENSEMBLE_LEARNING.EXPERT_GROUP:
                if self.cfg.ENSEMBLE_LEARNING.VOTING == 'C':
                    consensus = True
                elif self.cfg.ENSEMBLE_LEARNING.VOTING == 'U':
                    unanimous = True
            else:
                ensemble = True
        rel_binarys_matrix = None
        if boxes[0].has_field("relness_mat"):
            rel_binarys_matrix = [ each.get_field("relness_mat") for each in boxes]

            
        if self.attribute_on:
            if isinstance(refine_logits[0], (list, tuple)):
                finetune_obj_logits, finetune_att_logits = refine_logits
            else:
                # just use attribute feature, do not actually predict attribute
                self.attribute_on = False
                finetune_obj_logits = refine_logits
        else:
            finetune_obj_logits = refine_logits

        results = []

        if self.cfg.ENSEMBLE_LEARNING.EXPERT_GROUP:
            start = 0
            boxlist = None
            total_rel_det_num = rel_pair_idxs
            triple_scores_ensemble = []
            rel_pair_idx_ensemble = []
            rel_class_prob_ensemble = []
            rel_labels_ensemble = []
            start = 0
            chosen_labels_incr = []
            if 'group' in self.cfg.ENSEMBLE_LEARNING.TYPE and self.cfg.GLOBAL_SETTING.DATASET_CHOICE == 'VG':
                models = [['group_01', 'group_02', 'group_03'],  ['group_11', 'group_12', 'group_13'],
                          ['group_21', 'group_22', 'group_23'], ['group_31', 'group_32', 'group_33'],
                          ['group_41', 'group_42', 'group_43']]
                num_rel_classes = self.cfg.MODEL.ROI_RELATION_HEAD.VG_NUM_CLASSES
            elif 'group' in self.cfg.ENSEMBLE_LEARNING.TYPE and self.cfg.GLOBAL_SETTING.DATASET_CHOICE == 'GQA':
                models = [['group_01', 'group_02', 'group_03'],  ['group_11', 'group_12', 'group_13'],
                          ['group_21', 'group_22', 'group_23'], ['group_31', 'group_32', 'group_33']]
                num_rel_classes = self.cfg.MODEL.ROI_RELATION_HEAD.GQA_200_NUM_CLASSES
            else:
                models = self.cfg.ENSEMBLE_LEARNING.TYPE
            obj_logit = finetune_obj_logits[0]
            rel_pair_idx = rel_pair_idxs[0]
            box = boxes[0]
            if boxlist is None:
                obj_class_prob = F.softmax(obj_logit, -1)
                obj_class_prob[:, 0] = 0  # set background score to 0
                num_obj_bbox = obj_class_prob.shape[0]
                num_obj_class = obj_class_prob.shape[1]

                if self.use_gt_box:
                    obj_scores, obj_pred = obj_class_prob[:, 1:].max(dim=1)
                    obj_pred = obj_pred + 1
                else:
                    # NOTE: by kaihua, apply late nms for object prediction
                    obj_pred = obj_prediction_nms(box.get_field('boxes_per_cls'), obj_logit,
                                                  self.later_nms_pred_thres)
                    obj_score_ind = torch.arange(num_obj_bbox, device=obj_logit.device) * num_obj_class + obj_pred
                    obj_scores = obj_class_prob.view(-1)[obj_score_ind]

                assert obj_scores.shape[0] == num_obj_bbox
                obj_class = obj_pred

                if self.use_gt_box:
                    boxlist = box
                else:
                    # mode==sgdet
                    device = obj_class.device
                    batch_size = obj_class.shape[0]
                    regressed_box_idxs = obj_class
                    boxlist = BoxList(
                        box.get_field('boxes_per_cls')[torch.arange(batch_size, device=device), regressed_box_idxs],
                        box.size, 'xyxy')
                boxlist.add_field('pred_labels', obj_class)  # (#obj, )
                boxlist.add_field('pred_scores', obj_scores)  # (#obj, )

            # sorting triples according to score production
            obj_scores0 = obj_scores[rel_pair_idx[:, 0]]
            obj_scores1 = obj_scores[rel_pair_idx[:, 1]]
            for j, model in enumerate(models):
                rel_class_list = []
                rel_class_prob_list = []
                triple_scores_list = []
                chosen_idx_bool_list = []
                incre_idx_list_bool = [x == j + 1 for x in incre_idx_list]
                chosen_labels = [k for k, x in enumerate(incre_idx_list_bool) if x]
                chosen_labels_incr_ = [0] + chosen_labels
                chosen_labels_incr.append(chosen_labels_incr_)
                chosen_labels_ = [k + 1 for k in range(len(chosen_labels))]

                for i, ensemble_model in enumerate(model):
                    relation_logit = (relation_logits[ensemble_model],)

                    indices_list = []

                    for _, rel_logit in enumerate(relation_logit):

                        rel_class_prob = F.softmax(rel_logit, -1)
                        rel_class_prob = rel_class_prob[:, :-1]
                        rel_scores, rel_class = rel_class_prob[:, 1:].max(dim=1)
                        rel_class = rel_class + 1
                        t = [rel_class == k for k in chosen_labels_]
                        for k, val in enumerate(t):
                            if k == 0:
                                chosen_idx_bool = val
                            else:
                                chosen_idx_bool = torch.logical_or(chosen_idx_bool, val)
                        triple_scores_list.append(rel_scores * obj_scores0 * obj_scores1)
                        chosen_idx_bool_list.append(chosen_idx_bool)
                        rel_class_list.append(rel_class)
                        rel_class_prob_list.append(rel_class_prob)
                rel_class_tensor = []
                chosen_idx_bool1 = torch.eq(chosen_idx_bool_list[0], chosen_idx_bool_list[1])
                rel_class_tensor.append(torch.logical_and(torch.eq(rel_class_list[0], rel_class_list[1]), chosen_idx_bool1))


                chosen_idx_bool2 = torch.eq(chosen_idx_bool_list[1], chosen_idx_bool_list[2])
                rel_class_tensor.append(torch.logical_and(torch.eq(rel_class_list[1], rel_class_list[2]), chosen_idx_bool2))
                chosen_idx_bool3 = torch.eq(chosen_idx_bool_list[0], chosen_idx_bool_list[2])
                rel_class_tensor.append(torch.logical_and(torch.eq(rel_class_list[0], rel_class_list[2]), chosen_idx_bool3))

                triple_scores_avg_all = torch.mean(torch.cat((triple_scores_list[0].unsqueeze(1),
                                                              triple_scores_list[1].unsqueeze(1),
                                                              triple_scores_list[2].unsqueeze(1)), dim=1), dim=1)

                rel_class_prob_avg_all = torch.mean(torch.cat((rel_class_prob_list[0].unsqueeze(1),
                                                              rel_class_prob_list[1].unsqueeze(1),
                                                              rel_class_prob_list[2].unsqueeze(1)), dim=1), dim=1)


                if consensus: #majority must agree
                    triple_scores_avg1 = torch.mean(torch.cat((triple_scores_list[0].unsqueeze(1), triple_scores_list[1].unsqueeze(1)), dim=1), dim=1)
                    rel_class_prob_avg1 = torch.mean(
                        torch.cat((rel_class_prob_list[0].unsqueeze(1), rel_class_prob_list[1].unsqueeze(1)), dim=1),
                        dim=1)
                    triple_scores_avg2 = torch.mean(
                        torch.cat((triple_scores_list[1].unsqueeze(1), triple_scores_list[2].unsqueeze(1)), dim=1),
                        dim=1)
                    rel_class_prob_avg2 = torch.mean(
                        torch.cat((rel_class_prob_list[1].unsqueeze(1), rel_class_prob_list[1].unsqueeze(1)), dim=1),
                        dim=1)

                    triple_scores_avg3 = torch.mean(
                        torch.cat((triple_scores_list[0].unsqueeze(1), triple_scores_list[2].unsqueeze(1)), dim=1),
                        dim=1)
                    rel_class_prob_avg3 = torch.mean(
                        torch.cat((rel_class_prob_list[0].unsqueeze(1), rel_class_prob_list[2].unsqueeze(1)), dim=1),
                        dim=1)


                    rel_class_indices_bool = torch.cat((rel_class_tensor[0].unsqueeze(1), rel_class_tensor[1].unsqueeze(1), rel_class_tensor[2].unsqueeze(1)), dim=1)#torch.logical_or(torch.logical_or(rel_class_tensor1, rel_class_tensor2), rel_class_tensor3)
                    triple_scores_avg = torch.cat((triple_scores_avg1.unsqueeze(1), triple_scores_avg2.unsqueeze(1), triple_scores_avg3.unsqueeze(1)), dim=1)
                    rel_class_prob_avg = torch.cat((rel_class_prob_avg1.unsqueeze(1), rel_class_prob_avg2.unsqueeze(1), rel_class_prob_avg3.unsqueeze(1)), dim=1)
                    triple_scores_new = triple_scores_avg.clone()
                    triple_scores_new.masked_fill_(~rel_class_indices_bool, 0)

                    count = torch.sum(rel_class_indices_bool, dim=1)
                    triple_scores = torch.sum(triple_scores_new, dim=1) / count
                    triple_scores[torch.isnan(triple_scores)] = 0
                    rel_class_prob_new = rel_class_prob_avg
                    rel_class_prob_indices_bool = rel_class_indices_bool.unsqueeze(2).repeat(1, 1, len(chosen_labels_incr_))
                    rel_class_prob_new.masked_fill_(~rel_class_prob_indices_bool, 0)
                    count2 = count.unsqueeze(1)
                    count2 = count2.repeat(1, len(chosen_labels_incr_))
                    rel_class_prob = torch.sum(rel_class_prob_new, dim=1) / count2
                    rel_class_prob[torch.isnan(triple_scores)] = 0
                    rel_class = torch.zeros_like(rel_class_list[0])
                    for rel_c, rel_bool in zip(rel_class_list, rel_class_tensor):
                        rel_class[rel_bool.nonzero().squeeze(1).tolist()] = rel_c[rel_bool.nonzero().squeeze(1).tolist()]


                    rel_class_indices_bool = torch.logical_or(torch.logical_or(rel_class_tensor[0], rel_class_tensor[1]),
                                                               rel_class_tensor[2])

                elif unanimous: #all must agree
                    rel_class_indices_bool = torch.logical_and(torch.logical_and(rel_class_tensor[0], rel_class_tensor[1]),
                                                              rel_class_tensor[2])
                    triple_scores = triple_scores_avg_all
                    rel_class_prob = rel_class_prob_avg_all

                rel_class_indices = rel_class_indices_bool.nonzero().squeeze(1).tolist()
                triple_scores_ensemble.append(triple_scores[rel_class_indices].cuda())
                rel_pair_idx_ensemble.append(rel_pair_idx[rel_class_indices].cuda())
                rel_labels_ensemble.append(rel_class[rel_class_indices].cuda())
                rel_class_prob_ensemble.append(rel_class_prob[rel_class_indices].cuda())

            total_length = 0
            incr_length = []
            for scores in triple_scores_ensemble:
                total_length += len(scores)
                incr_length.append(len(scores))
            #triple_scores_final =
            triple_scores_final = torch.zeros(total_length).cuda()
            rel_pair_idx_final = torch.zeros(total_length, 2).cuda()
            rel_class_prob_final = torch.zeros(total_length, num_rel_classes).cuda()
            rel_labels_final = torch.zeros(total_length, dtype=torch.int64).cuda()
            start = 0
            for i, scores in enumerate(triple_scores_ensemble):
                triple_scores_final[start:start+incr_length[i]] = scores
                rel_pair_idx_final[start:start + incr_length[i]] = rel_pair_idx_ensemble[i]
                rel_class_prob_final[start:start + incr_length[i], chosen_labels_incr[i]] = rel_class_prob_ensemble[i]
                rel_labels_final[start:start + incr_length[i]] = rel_labels_ensemble[i]
                start += incr_length[i]
            _, sorting_idx = torch.sort(triple_scores_final.view(-1), dim=0, descending=True)
            rel_pair_idx_ensemble = rel_pair_idx_final[sorting_idx]
            rel_class_prob_ensemble = rel_class_prob_final[sorting_idx]
            rel_labels_ensemble = rel_labels_final[sorting_idx]
            boxlist.add_field('rel_pair_idxs', rel_pair_idx_ensemble)  # (#rel, 2)
            boxlist.add_field('pred_rel_scores', rel_class_prob_ensemble)  # (#rel, #rel_class)
            boxlist.add_field('pred_rel_labels', rel_labels_ensemble)  # (#rel, )
            results.append(boxlist)
        elif ensemble:
            start = 0
            boxlist = None
            total_rel_det_num = rel_pair_idxs
            triple_scores_ensemble = []
            rel_pair_idx_ensemble = []
            rel_class_prob_ensemble = []
            rel_labels_ensemble = []
            start = 0
            chosen_labels_incr = []
            if 'group' in self.cfg.ENSEMBLE_LEARNING.TYPE and self.cfg.GLOBAL_SETTING.DATASET_CHOICE == 'VG':
                models = ['group_0', 'group_1', 'group_2', 'group_3', 'group_4']
                num_rel_classes = self.cfg.MODEL.ROI_RELATION_HEAD.VG_NUM_CLASSES
            elif 'group' in self.cfg.ENSEMBLE_LEARNING.TYPE and self.cfg.GLOBAL_SETTING.DATASET_CHOICE == 'GQA':
                models = ['group_0', 'group_1', 'group_2', 'group_3']
                num_rel_classes = self.cfg.MODEL.ROI_RELATION_HEAD.GQA_200_NUM_CLASSES
            else:
                models = self.cfg.ENSEMBLE_LEARNING.TYPE
            for i, ensemble_model in enumerate(models):
                relation_logit = (relation_logits[ensemble_model],)
                for _, (rel_logit, obj_logit, rel_pair_idx, box) in enumerate(zip(
                        relation_logit, finetune_obj_logits, rel_pair_idxs, boxes
                )):

                    if boxlist is None:
                        if self.attribute_on:
                            att_logit = finetune_att_logits[i]
                            att_prob = torch.sigmoid(att_logit)
                        obj_class_prob = F.softmax(obj_logit, -1)
                        obj_class_prob[:, 0] = 0  # set background score to 0
                        num_obj_bbox = obj_class_prob.shape[0]
                        num_obj_class = obj_class_prob.shape[1]

                        if self.use_gt_box:
                            obj_scores, obj_pred = obj_class_prob[:, 1:].max(dim=1)
                            obj_pred = obj_pred + 1
                        else:
                            # NOTE: by kaihua, apply late nms for object prediction
                            obj_pred = obj_prediction_nms(box.get_field('boxes_per_cls'), obj_logit,
                                                          self.later_nms_pred_thres)
                            obj_score_ind = torch.arange(num_obj_bbox, device=obj_logit.device) * num_obj_class + obj_pred
                            obj_scores = obj_class_prob.view(-1)[obj_score_ind]

                        assert obj_scores.shape[0] == num_obj_bbox
                        obj_class = obj_pred

                        if self.use_gt_box:
                            boxlist = box
                        else:
                            # mode==sgdet
                            # apply regression based on finetuned object class
                            device = obj_class.device
                            batch_size = obj_class.shape[0]
                            regressed_box_idxs = obj_class
                            boxlist = BoxList(
                                box.get_field('boxes_per_cls')[torch.arange(batch_size, device=device), regressed_box_idxs],
                                box.size, 'xyxy')
                        boxlist.add_field('pred_labels', obj_class)  # (#obj, )
                        boxlist.add_field('pred_scores', obj_scores)  # (#obj, )

                        if self.attribute_on:
                            boxlist.add_field('pred_attributes', att_prob)

                    # sorting triples according to score production
                    obj_scores0 = obj_scores[rel_pair_idx[:, 0]]
                    obj_scores1 = obj_scores[rel_pair_idx[:, 1]]
                    rel_class_prob = F.softmax(rel_logit, -1)
                    rel_class_prob = rel_class_prob[:, :-1]
                    rel_scores, rel_class = rel_class_prob[:, 1:].max(dim=1) #background is filtered
                    rel_class = rel_class + 1 #background is filtered
                    incre_idx_list_bool = [x == i + 1 for x in incre_idx_list]
                    chosen_labels = [i for i, x in enumerate(incre_idx_list_bool) if x]
                    chosen_labels_incr_ = [0] + chosen_labels
                    chosen_labels_incr.append(chosen_labels_incr_)
                    chosen_labels_ = [i+1 for i in range(len(chosen_labels))]
                    t = [rel_class == i for i in chosen_labels_]


                    for i, val in enumerate(t):
                        if i == 0:
                            chosen_idx_bool = val
                        else:
                            chosen_idx_bool = torch.logical_or(chosen_idx_bool, val)
                    triple_scores = rel_scores * obj_scores0 * obj_scores1
                    indices = chosen_idx_bool.nonzero().squeeze(1).tolist()
                    triple_scores_ensemble.append(triple_scores[indices].cuda())
                    rel_pair_idx_ensemble.append(rel_pair_idx[indices].cuda())
                    rel_labels_ensemble.append(rel_class[indices].cuda())
                    rel_class_prob_ensemble.append(rel_class_prob[indices].cuda())

            total_length = 0
            incr_length = []
            for scores in triple_scores_ensemble:
                total_length += len(scores)
                incr_length.append(len(scores))
            triple_scores_final = torch.zeros(total_length).cuda()
            rel_pair_idx_final = torch.zeros(total_length, 2).cuda()
            rel_class_prob_final = torch.zeros(total_length, num_rel_classes).cuda()
            rel_labels_final = torch.zeros(total_length, dtype=torch.int64).cuda()
            start = 0
            for i, scores in enumerate(triple_scores_ensemble):
                triple_scores_final[start:start+incr_length[i]] = scores
                rel_pair_idx_final[start:start + incr_length[i]] = rel_pair_idx_ensemble[i]
                rel_class_prob_final[start:start + incr_length[i], chosen_labels_incr[i]] = rel_class_prob_ensemble[i]
                rel_labels_final[start:start + incr_length[i]] = rel_labels_ensemble[i]
                start += incr_length[i]
            _, sorting_idx = torch.sort(triple_scores_final.view(-1), dim=0, descending=True)
            rel_pair_idx_ensemble = rel_pair_idx_final[sorting_idx]
            rel_class_prob_ensemble = rel_class_prob_final[sorting_idx]
            rel_labels_ensemble = rel_labels_final[sorting_idx]
            boxlist.add_field('rel_pair_idxs', rel_pair_idx_ensemble)  # (#rel, 2)
            boxlist.add_field('pred_rel_scores', rel_class_prob_ensemble)  # (#rel, #rel_class)
            boxlist.add_field('pred_rel_labels', rel_labels_ensemble)  # (#rel, )
            results.append(boxlist)
        else:
            for i, (rel_logit, obj_logit, rel_pair_idx, box) in enumerate(zip(
                relation_logits, finetune_obj_logits, rel_pair_idxs, boxes
            )):
                if self.attribute_on:
                    att_logit = finetune_att_logits[i]
                    att_prob = torch.sigmoid(att_logit)
                obj_class_prob = F.softmax(obj_logit, -1)
                obj_class_prob[:, 0] = 0  # set background score to 0
                num_obj_bbox = obj_class_prob.shape[0]
                num_obj_class = obj_class_prob.shape[1]

                if self.use_gt_box:
                    obj_scores, obj_pred = obj_class_prob[:, 1:].max(dim=1)
                    obj_pred = obj_pred + 1
                else:
                    # NOTE: by kaihua, apply late nms for object prediction
                    obj_pred = obj_prediction_nms(box.get_field('boxes_per_cls'), obj_logit, self.later_nms_pred_thres)
                    obj_score_ind = torch.arange(num_obj_bbox, device=obj_logit.device) * num_obj_class + obj_pred
                    obj_scores = obj_class_prob.view(-1)[obj_score_ind]

                assert obj_scores.shape[0] == num_obj_bbox
                obj_class = obj_pred

                if self.use_gt_box:
                    boxlist = box
                else:
                    # mode==sgdet
                    # apply regression based on finetuned object class
                    device = obj_class.device
                    batch_size = obj_class.shape[0]
                    regressed_box_idxs = obj_class
                    boxlist = BoxList(box.get_field('boxes_per_cls')[torch.arange(batch_size, device=device), regressed_box_idxs], box.size, 'xyxy')
                boxlist.add_field('pred_labels', obj_class) # (#obj, )
                boxlist.add_field('pred_scores', obj_scores) # (#obj, )

                if self.attribute_on:
                    boxlist.add_field('pred_attributes', att_prob)

                # sorting triples according to score production
                obj_scores0 = obj_scores[rel_pair_idx[:, 0]]
                obj_scores1 = obj_scores[rel_pair_idx[:, 1]]
                rel_class_prob = F.softmax(rel_logit, -1)
                rel_scores, rel_class = rel_class_prob[:, 1:].max(dim=1)
                rel_class = rel_class + 1

                triple_scores = rel_scores * obj_scores0 * obj_scores1
                _, sorting_idx = torch.sort(triple_scores.view(-1), dim=0, descending=True)
                rel_pair_idx = rel_pair_idx[sorting_idx]
                rel_class_prob = rel_class_prob[sorting_idx]
                rel_labels = rel_class[sorting_idx]

                boxlist.add_field('rel_pair_idxs', rel_pair_idx)
                boxlist.add_field('pred_rel_scores', rel_class_prob)
                boxlist.add_field('pred_rel_labels', rel_labels)
                results.append(boxlist)
        return results


def make_roi_relation_post_processor(cfg):
    attribute_on = cfg.MODEL.ATTRIBUTE_ON
    use_gt_box = cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX
    later_nms_pred_thres = cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES

    postprocessor = PostProcessor(
        attribute_on,
        use_gt_box,
        later_nms_pred_thres,
        cfg,
    )
    return postprocessor
