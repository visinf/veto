# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn


from pysgg.config import cfg
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
            later_nms_pred_thres=0.3,
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
        ensemble_final = False

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
        if ensemble_final:
            #with open('/data/user/models/GCL/test/inference/VG_stanford_filtered_with_attribute_incl_depth_test/mR100_per_model.pkl', 'rb') as file:
            #    mR_100 = pickle.load(file)
            #expert = [0] * 50
            #m1 = list(mR_100['gcl'].values())
            #m2 = list(mR_100['beta'].values())
            #m3 = list(mR_100['vanilla'].values())
            #for i, val in enumerate(zip(m1, m2, m3)):
            #    val = np.array(val)
             #   expert[i] = np.argmax(val)
            rel_det_num = len(rel_pair_idxs[0])
            triple_scores_ensemble = torch.zeros(rel_det_num*2)
            rel_pair_idx_ensemble = torch.zeros(rel_det_num*2, 2)
            rel_class_prob_ensemble = torch.zeros(rel_det_num*2, 51)
            rel_labels_ensemble = torch.zeros(rel_det_num*2, dtype=torch.int64)
            start = 0
            for ensemble_model in ['gcl', 'vanilla']: #relation_logits.keys():
                relation_logit = (relation_logits[ensemble_model],)
                for i, (rel_logit, obj_logit, rel_pair_idx, box) in enumerate(zip(
                        relation_logit, finetune_obj_logits, rel_pair_idxs, boxes
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
                    rel_class_prob_ensemble[start:start+rel_det_num] = rel_class_prob
                    rel_scores, rel_class = rel_class_prob[:, 1:].max(dim=1)
                    #temp = entropy(rel_class_prob.cpu(), base=2, axis=1)
                    #temp = torch.Tensor(temp).cuda()
                    triple_scores_ensemble[start:start+rel_det_num] = rel_scores * obj_scores0 * obj_scores1   #- temp #entropy(rel_scores, base=2)
                    rel_pair_idx_ensemble[start:start+rel_det_num] = rel_pair_idx
                    rel_labels_ensemble[start:start+rel_det_num] = rel_class
                    #rel_scores, rel_class = rel_class_prob[:, 1:].max(dim=1)
                    #rel_class = rel_class + 1

                    # TODO Kaihua: how about using weighted some here?  e.g. rel*1 + obj *0.8 + obj*0.8
                    #triple_scores = rel_scores * obj_scores0 * obj_scores1
                    #_, sorting_idx = torch.sort(triple_scores.view(-1), dim=0, descending=True)
                    #rel_pair_idx = rel_pair_idx[sorting_idx]
                    #rel_class_prob = rel_class_prob[sorting_idx]
                    #rel_labels = rel_class[sorting_idx]

                    #boxlist.add_field('rel_pair_idxs_%s' % ensemble_model, rel_pair_idx)  # (#rel, 2)
                    #boxlist.add_field('pred_rel_scores_%s' % ensemble_model, rel_class_prob)  # (#rel, #rel_class)
                    #boxlist.add_field('pred_rel_labels_%s' % ensemble_model, rel_labels)  # (#rel, )
                    # should have fields : rel_pair_idxs, pred_rel_class_prob, pred_rel_labels, pred_labels, pred_scores
                    # Note
                    # TODO Kaihua: add a new type of element, which can have different length with boxlist (similar to field, except that once
                    # the boxlist has such an element, the slicing operation should be forbidden.)
                    # it is not safe to add fields about relation into boxlist!
                    start += rel_det_num
            _, sorting_idx = torch.sort(triple_scores_ensemble.view(-1), dim=0, descending=True)
            rel_pair_idx_ensemble = rel_pair_idx_ensemble[sorting_idx]
            rel_class_prob_ensemble = rel_class_prob_ensemble[sorting_idx]
            rel_labels_ensemble = rel_labels_ensemble[sorting_idx]
            boxlist.add_field('rel_pair_idxs', rel_pair_idx_ensemble)  # (#rel, 2)
            boxlist.add_field('pred_rel_scores', rel_class_prob_ensemble)  # (#rel, #rel_class)
            boxlist.add_field('pred_rel_labels', rel_labels_ensemble)  # (#rel, )
            """
            rel_class_prob_ensemble = torch.zeros(boxlist.extra_fields['pred_rel_scores_gcl'].size())
            for i in range(boxlist.extra_fields['pred_rel_scores_gcl'].size(0)):
                for j in range(boxlist.extra_fields['pred_rel_scores_gcl'].size(1)):
                    val = [boxlist.extra_fields['pred_rel_scores_gcl'][i][j], boxlist.extra_fields['pred_rel_scores_beta'][i][j], boxlist.extra_fields['pred_rel_scores_vanilla'][i][j]]
                    if j == 0:
                        rel_class_prob_ensemble[i][j] = sum(val)/3
                    else:
                        rel_class_prob_ensemble[i][j] = val[expert[j-1]]
            _, pred_rel_labels_ensemble = rel_class_prob_ensemble[:, 1:].max(dim=1)
            pred_rel_labels_ensemble += 1
            """
            #boxlist.add_field('pred_rel_labels', pred_rel_labels_ensemble)
            #boxlist.add_field('pred_rel_scores', rel_class_prob_ensemble)
            results.append(boxlist)
        elif ensemble:
            #rel_det_num = len(rel_pair_idxs[0])
            start = 0
            boxlist = None
            total_rel_det_num = rel_pair_idxs
            triple_scores_ensemble = [] #torch.zeros(rel_det_num * 2)
            rel_pair_idx_ensemble = []  #torch.zeros(rel_det_num * 2, 2)
            rel_class_prob_ensemble = [] #torch.zeros(rel_det_num * 2, 51)
            rel_labels_ensemble = [] #torch.zeros(rel_det_num * 2, dtype=torch.int64)
            start = 0
            chosen_labels_incr = []
            if 'group' in self.cfg.ENSEMBLE_LEARNING.TYPE:
                models = ['group_0', 'group_1', 'group_2', 'group_3', 'group_4']
            else:
                models = self.cfg.ENSEMBLE_LEARNING.TYPE
            for i, ensemble_model in enumerate(models):  #self.cfg.ENSEMBLE_LEARNING.TYPE:  # relation_logits.keys():
                #rel_pair_idx = rel_pair_idx[cur_chosen_matrix[i]]
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
                    rel_scores, rel_class = rel_class_prob[:, 1:].max(dim=1)
                    rel_class = rel_class + 1
                    incre_idx_list_bool = [x == i + 1 for x in incre_idx_list]
                    chosen_labels = [i for i, x in enumerate(incre_idx_list_bool) if x]
                    chosen_labels_incr_ = [0] + chosen_labels
                    chosen_labels_incr.append(chosen_labels_incr_)
                    chosen_labels_ = [i+1 for i in range(len(chosen_labels))]
                    chosen_idx = [i for i, x in enumerate(rel_class) if x in chosen_labels_]
                    # TODO Kaihua: how about using weighted some here?  e.g. rel*1 + obj *0.8 + obj*0.8
                    triple_scores = rel_scores * obj_scores0 * obj_scores1
                    triple_scores_ensemble.append([i.item() for i, x in zip(triple_scores, chosen_idx) if x])  # - temp #entropy(rel_scores, base=2)
                    rel_pair_idx_ensemble.append([[i[0].item(), i[1].item()] for i, x in zip(rel_pair_idx, chosen_idx) if x])
                    rel_labels_ensemble.append([i.item() for i, x in zip(rel_class, chosen_idx) if x])
                    rel_class_prob_ensemble.append([[j.item() for j in i] for i, x in zip(rel_class_prob, chosen_idx) if x])

            total_length = 0
            incr_length = []
            for scores in triple_scores_ensemble:
                total_length += len(scores)
                incr_length.append(len(scores))
            #triple_scores_final =
            triple_scores_final = torch.zeros(total_length)
            rel_pair_idx_final = torch.zeros(total_length, 2)
            rel_class_prob_final = torch.zeros(total_length, 51)
            rel_labels_final = torch.zeros(total_length, dtype=torch.int64)

            start = 0
            for i, scores in enumerate(triple_scores_ensemble):
                triple_scores_final[start:start+incr_length[i]] = torch.tensor(scores)
                rel_pair_idx_final[start:start + incr_length[i]] = torch.tensor(rel_pair_idx_ensemble[i])
                rel_class_prob_final[start:start + incr_length[i], chosen_labels_incr[i]] = torch.tensor(rel_class_prob_ensemble[i])
                rel_labels_final[start:start + incr_length[i]] = torch.tensor(rel_labels_ensemble[i])
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
                # TODO Kaihua: how about using weighted some here?  e.g. rel*1 + obj *0.8 + obj*0.8
                triple_scores = rel_scores * obj_scores0 * obj_scores1
                _, sorting_idx = torch.sort(triple_scores.view(-1), dim=0, descending=True)
                rel_pair_idx = rel_pair_idx[sorting_idx]
                rel_class_prob = rel_class_prob[sorting_idx]
                rel_labels = rel_class[sorting_idx]

                boxlist.add_field('rel_pair_idxs', rel_pair_idx) # (#rel, 2)
                boxlist.add_field('pred_rel_scores', rel_class_prob) # (#rel, #rel_class)
                boxlist.add_field('pred_rel_labels', rel_labels) # (#rel, )
                # should have fields : rel_pair_idxs, pred_rel_class_prob, pred_rel_labels, pred_labels, pred_scores
                # Note
                # TODO Kaihua: add a new type of element, which can have different length with boxlist (similar to field, except that once
                # the boxlist has such an element, the slicing operation should be forbidden.)
                # it is not safe to add fields about relation into boxlist!
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
    )
    return postprocessor
