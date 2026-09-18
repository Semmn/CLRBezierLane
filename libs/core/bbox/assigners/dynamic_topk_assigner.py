# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmdet.models.task_modules.assigners.base_assigner import BaseAssigner
from mmdet.registry import TASK_UTILS

@TASK_UTILS.register_module()
class DynamicTopkAssigner(BaseAssigner):
    """Computes dynamick-to-one lane matching between predictions and ground truth (GT).
    The dynamic k for each GT is computed using Lane(Line)IoU matrix.
    The costs matrix is calculated from:
    1) CLRNet: lane horizontal distance, starting point xy, angle and classification scores.
    2) CLRerNet: LaneIoU and classification scores.
    After the dynamick-to-one matching, the un-matched priors are treated as backgrounds.
    Thus each prior's prediction will be assigned with `0` or a positive integer
    indicating the ground truth index:
    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt
    Args:
        cls_cost (dict): cls cost config
        iou_dynamick (dict): iou cost config for dynamic-k calculation
        iou_cost (dict): iou cost config
        reg_cost (dict): reg cost config
        reg_weight (float): cost weight for regression
        cost_combination (int): cost calculation type. 0: CLRNet, 1: CLRerNet.
        use_pred_length_for_iou (bool): prepare pred lane length for iou calculation.
        max_topk (int): max value for dynamic-k.
        min_topk (int): min value for dynamic-k.
    """

    def __init__(
        self,
        cls_cost=None,
        iou_dynamick=None,
        iou_cost=None,
        reg_cost=None,
        reg_weight=3.0,
        cost_combination=1,
        assignment_type=0,
        use_pred_length_for_iou=True,
        max_topk=4,
        min_topk=1,
        use_dynamick_alt=True
    ):
        self.cls_cost = TASK_UTILS.build(cls_cost)
        self.reg_cost = TASK_UTILS.build(reg_cost)
        self.iou_dynamick = TASK_UTILS.build(iou_dynamick)
        self.iou_cost = TASK_UTILS.build(iou_cost)
        self.use_pred_length_for_iou = use_pred_length_for_iou
        self.max_topk = max_topk
        self.min_topk = min_topk
        self.reg_weight = reg_weight
        self.cost_combination = cost_combination
        self.assignment_type = assignment_type  # 0: dynamic k assign, 1: atss assign, 2: atss assign2
        # if use_dynamick_alt=True, self.min_topk is used as the minimum allocation if the atss assignment fails.
        self.use_dynamick_alt = use_dynamick_alt # use dynamic k assignment alternative when using ATSS assignment
        
    
    def atss_assign2(self, cost, ious_matrix):
        """"
        Same as atss_assign1, but first selects top_k candidates from the cost matrix.
        direct assignment are based on iou_matrix calculation.
        
        to prevent none of priors are matched with ground truths, dynamic_k assignment is used for the case.
        it guarantees that at least min_topk priors are matched with each ground truth.
        
        Args:
            cost: the assign cost, shape (Np, Ng).
            ious_matrix: iou of grouth truth and priors, shape (Np, Ng).
        Returns:
            torch.Tensor: the indices of assigned prior.
            torch.Tensor: the corresponding ground truth indices.
        Np: number of priors (anchors), Ng: number of GT lanes.
        """
        matching_matrix = torch.zeros_like(cost) # (Np, Ng)
        ious_matrix[ious_matrix < 0] = 0.0 # avoid negative iou values - (Np, Ng)
        
        if self.use_dynamick_alt:
            topk_ious, _ = torch.topk(ious_matrix, self.max_topk, dim=0)
            dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=self.min_topk)
        
        # 1. select top_k largest iou values from the iou_matrix (select the candidates)
        topk_cost, topk_idxes = torch.topk(cost, self.max_topk, dim=0, largest=False) # select top_k largest iou values from the iou_matrix - (Np, Ng)->(max_topk, Ng)
        candidates = torch.gather(ious_matrix, dim=0, index=topk_idxes)
        
        # 2. calculate the mean and variation of the top_k cost values
        ious_mean, ious_val = torch.mean(candidates, dim=0), \
            torch.std(candidates, dim=0) # (max_topk, Ng) -> (Ng,)
        
        # 3. calculate the dynamic threshold for positive samples
        # thresholds are calculated independently for each ground truth
        threshold = ious_mean + ious_val # calculate the threshold for positive samples - (Ng, )
        
        num_gt = ious_matrix.shape[1] # number of ground truths (Ng)
        for gt_idx in range(num_gt): # iterate through each ground truth
            pos_idx = ious_matrix[:, gt_idx] >= threshold[gt_idx] # for each ground truth, find the priors (predictions) with iou higher than the threshold
            
            if pos_idx.sum() == 0:
                if self.use_dynamick_alt:
                    pos_idx = torch.topk(ious_matrix[:, gt_idx], k=dynamic_ks[gt_idx], largest=True)[1] 
                else:
                    pos_idx = torch.topk(ious_matrix[:, gt_idx], k=self.min_topk, largest=True)[1] # select top_k largest iou values from the iou_matrix
            
            matching_matrix[pos_idx, gt_idx] = 1.0 # update matching matrix with 1.0 for the selected priors (predictions) for each gt
        del topk_cost, threshold, pos_idx # release the memory of topk_ious, threshold

        matched_gt = matching_matrix.sum(1) # count the number of matched ground truths for each prior (anchor) - (Np, )
        if (matched_gt > 1).sum() > 0: # if there is at least one prior matched with multiple ground truths
            _, iou_argmax = torch.max(ious_matrix[matched_gt > 1, :], dim=1) # find the index of the highest iou for those priors
            matching_matrix[matched_gt > 1, 0] *= 0.0 # reset the matching matrix for those priors with non highest iou to 0.0
            matching_matrix[matched_gt > 1, iou_argmax] = 1.0 # allow only one prior to be matched with one groudn truth (lowest cost will be selected)

        prior_idx = matching_matrix.sum(1).nonzero() # find the indices of the matched prior only with at least one matched ground truth
        gt_idx = matching_matrix[prior_idx].argmax(-1) # find the corresponding ground truth indices for the matched priors
        return prior_idx.flatten(), gt_idx.flatten()
    
        
    def atss_assign(self, cost, ious_matrix):
        """"
        Assign groud truths with priors using ATSS method.
        unlike original ATSS method, in the lane detection task, 
        candidate priors are sampled by the fixed top-k iou values.
        then from the candidate prios, mean and valiation of iou, cost values are calculated
        and threshold for positive will be calculated. the notation and indexing follows the clrernet framework.
        
        to prevent none of priors are matched with ground truths, dynamic_k assignment is used for the case.
        it guarantees that at least min_topk priors are matched with each ground truth.
        
        Args:
            cost: the assign cost, shape (Np, Ng).
            ious_matrix: iou of grouth truth and priors, shape (Np, Ng).
        Returns:
            torch.Tensor: the indices of assigned prior.
            torch.Tensor: the corresponding ground truth indices.
        Np: number of priors (anchors), Ng: number of GT lanes.
        """
        matching_matrix = torch.zeros_like(cost) # (Np, Ng)
        ious_matrix[ious_matrix < 0] = 0.0 # avoid negative iou values - (Np, Ng)
        
        # 1. select top_k largest iou values from the iou_matrix (select the candidates)
        topk_ious, topk_idxes = torch.topk(ious_matrix, self.max_topk, dim=0) # select top_k largest iou values from the iou_matrix - (Np, Ng)->(max_topk, Ng)
        if self.use_dynamick_alt:
            dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=self.min_topk)
        candidates = torch.gather(ious_matrix, dim=0, index=topk_idxes)
        
        # 2. calculate the mean and variation of the top_k cost values
        cost_mean, cost_val = torch.mean(candidates, dim=0),\
            torch.std(candidates, dim=0) # (max_topk, Ng) -> (Ng,)
        
        # 3. calculate the dynamic threshold for positive samples
        # thresholds are calculated independently for each ground truth
        # unlike original ATSS, the lower cost means the better matching, so we abstract the cost_mean and cost_val
        threshold = cost_mean - cost_val # calculate the threshold for positive samples - (Ng, )
        
        num_gt = cost.shape[1] # number of ground truths (Ng)
        for gt_idx in range(num_gt): # iterate through each ground truth
            pos_idx = cost[:, gt_idx] <= threshold[gt_idx] # for each ground truth, find the priors (predictions) with cost lower than the threshold
            
            if pos_idx.sum() == 0:
                if self.use_dynamick_alt:
                    pos_idx = torch.topk(cost[:, gt_idx], k=dynamic_ks[gt_idx], largest=False)[1]
                else:
                    pos_idx = torch.topk(cost[:, gt_idx], k=self.min_topk, largest=False)[1]
                
            matching_matrix[pos_idx, gt_idx] = 1.0 # update matching matrix with 1.0 for the selected priors (predictions) for each gt
        del topk_ious, threshold, pos_idx # release the memory of topk_ious, threshold

        matched_gt = matching_matrix.sum(1) # count the number of matched ground truths for each prior (anchor) - (Np, )
        if (matched_gt > 1).sum() > 0: # if there is at least one prior matched with multiple ground truths
            _, cost_argmin = torch.min(cost[matched_gt > 1, :], dim=1) # find the index of the minimum cost for those priors
            matching_matrix[matched_gt > 1, 0] *= 0.0 # reset the matching matrix for those priors with non lowest cost to 0.0
            matching_matrix[matched_gt > 1, cost_argmin] = 1.0 # allow only one prior to be matched with one groudn truth (lowest cost will be selected)

        prior_idx = matching_matrix.sum(1).nonzero() # find the indices of the matched prior only with at least one matched ground truth
        gt_idx = matching_matrix[prior_idx].argmax(-1) # find the corresponding ground truth indices for the matched priors
        return prior_idx.flatten(), gt_idx.flatten()
        

    # simOTA assignment algorithm
    def dynamic_k_assign(self, cost, ious_matrix):
        """
        Assign grouth truths with priors dynamically.
        Args:
            cost: the assign cost, shape (Np, Ng).
            ious_matrix: iou of grouth truth and priors, shape (Np, Ng).
        Returns:
            torch.Tensor: the indices of assigned prior.
            torch.Tensor: the corresponding ground truth indices.
        Np: number of priors (anchors), Ng: number of GT lanes.
        """
        matching_matrix = torch.zeros_like(cost) # (Np, Ng)
        ious_matrix[ious_matrix < 0] = 0.0 # avoid negative iou values - (Np, Ng)
        topk_ious, _ = torch.topk(ious_matrix, self.max_topk, dim=0) # select top_k largest iou values from the iou_matrix - (Np, Ng)->(max_topk, Ng)
        
        #####################################################
        # track this: num_assignment, total_assignment
        # exact number of assignment by IoU
 
        # sum the top_k iou values for each gt, and clamp to min_topk - (Ng, )
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=self.min_topk) # min_topk=1 by default
        
        num_assignment = topk_ious.sum(0).int().sum()
        total_assignment =  dynamic_ks.sum()
        #####################################################
        
        num_gt = cost.shape[1] # number of ground truths (Ng)
        for gt_idx in range(num_gt): # iterate through each ground truth
            _, pos_idx = torch.topk(
                cost[:, gt_idx], k=dynamic_ks[gt_idx].item(), largest=False # find the indices of the lowest cost for each gt - (dynamic_ks[gt_idx] <= Np)
            )
            matching_matrix[pos_idx, gt_idx] = 1.0 # update matching matrix with 1.0 for the selected priors (predictions) for each gt
        del topk_ious, dynamic_ks, pos_idx # release the memory of topk_ious, dynamic_ks, pos_idx

        matched_gt = matching_matrix.sum(1) # count the number of matched ground truths for each prior (anchor) - (Np, )
        if (matched_gt > 1).sum() > 0: # if there is at least one prior matched with multiple ground truths
            _, cost_argmin = torch.min(cost[matched_gt > 1, :], dim=1) # find the index of the minimum cost for those priors
            matching_matrix[matched_gt > 1, 0] *= 0.0 # reset the matching matrix for those priors with non lowest cost to 0.0
            matching_matrix[matched_gt > 1, cost_argmin] = 1.0 # allow only one prior to be matched with one groudn truth (lowest cost will be selected)

        prior_idx = matching_matrix.sum(1).nonzero() # find the indices of the matched prior only with at least one matched ground truth
        gt_idx = matching_matrix[prior_idx].argmax(-1) # find the corresponding ground truth indices for the matched priors
        
        return prior_idx.flatten(), gt_idx.flatten(), num_assignment, total_assignment

    def _clrnet_cost(self, predictions, targets, pred_xs, target_xs, img_w, img_h):
        """_summary_
        Adapted from:
        https://github.com/Turoad/CLRNet/blob/main/clrnet/models/utils/dynamic_assign.py
        Args:
            predictions (Dict[torch.Trnsor]): predictions predicted by each stage, including:
                cls_logits: shape (Np, 2), anchor_params: shape (Np, 3),
                lengths: shape (Np, 1) and xs: shape (Np, Nr).
            targets (torch.Tensor): lane targets, shape: (Ng, 6+Nr).
                The first 6 elements are classification targets (2 ch), anchor starting point xy (2 ch),
                anchor theta (1ch) and anchor length (1ch).
            pred_xs (torch.Tensor): predicted x-coordinates on the predefined rows, shape (Np, Nr).
            target_xs (torch.Tensor): GT x-coordinates on the predefined rows, shape (Ng, Nr).
            img_w (int): network input image width (after crop and resize).
            img_h (int): network input image height (after crop and resize).
        Np: number of priors (anchors), Ng: number of GT lanes, Nr: number of rows.

        Returns:
            torch.Tensor: cost matrix, shape (Np, Ng).
        """
        num_priors = predictions["cls_logits"].shape[0] # number of priors (anchors)
        num_targets = targets.shape[0] # number of ground truths (Ng)
        
        # calculate distances cost (1. distance cost is considered. the distance between start point of anchor)
        distances_score = self.reg_cost(pred_xs, target_xs)
        distances_score = (
            1 - (distances_score / torch.max(distances_score)) + 1e-2 
        )  # normalize the distance

        target_start_xys = targets[:, 2:4]  # num_targets, 2
        target_start_xys[..., 0] *= img_h - 1 # unnormalize y-coordinate
        pred_reg_params = predictions["anchor_params"].detach().clone() # (Np, 3)
        pred_reg_params[:, 0] *= img_h - 1 # unnormalize y-coordinate
        pred_reg_params[:, 1] *= img_w - 1 # unnormalize x-coordinate
        start_xys_score = torch.cdist( # calculate the distance between start point of anchor and ground truth
            pred_reg_params[:, :2], target_start_xys, p=2
        ).reshape(num_priors, num_targets) # (Np, Ng)
        start_xys_score = (1 - start_xys_score / torch.max(start_xys_score)) + 1e-2

        pred_thetas = pred_reg_params[:, 2:3]  # (192, 1)
        target_thetas = targets[:, 4:5]  # (4, 1)
        theta_score = ( # calculate the angle difference between anchor and ground truth
            torch.cdist(pred_thetas, target_thetas, p=1).reshape(
                num_priors, num_targets
            )
            * 180 # (Np, Ng)
        )
        theta_score = (1 - theta_score / torch.max(theta_score)) + 1e-2

        # classification cost
        cls_score = self.cls_cost(
            predictions["cls_logits"].detach().clone(), targets[:, 1].long()
        ) # (Np, Ng)

        # ClrNet calculates the distance cost, start point xy cost, theta cost and classification cost
        cost = (
            -((distances_score * start_xys_score * theta_score) ** 2) * self.reg_weight
            + cls_score
        )
        return cost

    def _clrernet_cost(self, predictions, targets, pred_xs, target_xs):
        """_summary_

        Args:
            predictions (Dict[torch.Trnsor]): predictions predicted by each stage, including:
                cls_logits: shape (Np, 2), anchor_params: shape (Np, 3),
                lengths: shape (Np, 1) and xs: shape (Np, Nr).
            targets (torch.Tensor): lane targets, shape: (Ng, 6+Nr).
                The first 6 elements are classification targets (2 ch), anchor starting point xy (2 ch),
                anchor theta (1ch) and anchor length (1ch).
            pred_xs (torch.Tensor): predicted x-coordinates on the predefined rows, shape (Np, Nr).
            target_xs (torch.Tensor): GT x-coordinates on the predefined rows, shape (Ng, Nr).

        Returns:
            torch.Tensor: cost matrix, shape (Np, Ng).
        Np: number of priors (anchors), Ng: number of GT lanes, Nr: number of rows.
        """
        start = end = None
        if self.use_pred_length_for_iou:
            y0 = predictions["anchor_params"][:, 0].detach().clone()
            length = predictions["lengths"][:, 0].detach().clone()
            start = (1 - y0).clamp(min=0, max=1)
            end = (start + length).clamp(min=0, max=1)
        iou_cost = self.iou_cost(
            pred_xs, # predicted x-coordinates on the predefined rows, shape (Np, Nr)
            target_xs, # GT x-coordinates on the predefined rows, shape (Ng, Nr)
            start,
            end,
        )
        iou_score = 1 - (1 - iou_cost) / torch.max(1 - iou_cost) + 1e-2
        # classification cost
        cls_score = self.cls_cost(
            predictions["cls_logits"].detach().clone(), targets[:, 1].long()
        )
        # instead CLRerNet calculates the iou cost and classification cost
        cost = -iou_score * self.reg_weight + cls_score
        return cost

    def _clrernet_cost_wo_cls(self, predictions, pred_xs, target_xs):
        """_summary_
        clrernet_cost without classification cost
        Args:
            predictions (Dict[torch.Trnsor]): predictions predicted by each stage, including:
                cls_logits: shape (Np, 2), anchor_params: shape (Np, 3),
                lengths: shape (Np, 1) and xs: shape (Np, Nr).
            pred_xs (torch.Tensor): predicted x-coordinates on the predefined rows, shape (Np, Nr).
            target_xs (torch.Tensor): GT x-coordinates on the predefined rows, shape (Ng, Nr).

        Returns:
            torch.Tensor: cost matrix, shape (Np, Ng).
        Np: number of priors (anchors), Ng: number of GT lanes, Nr: number of rows.
        """
        start = end = None
        if self.use_pred_length_for_iou:
            y0 = predictions["anchor_params"][:, 0].detach().clone()
            length = predictions["lengths"][:, 0].detach().clone()
            start = (1 - y0).clamp(min=0, max=1)
            end = (start + length).clamp(min=0, max=1)
        iou_cost = self.iou_cost(
            pred_xs, # predicted x-coordinates on the predefined rows, shape (Np, Nr)
            target_xs, # GT x-coordinates on the predefined rows, shape (Ng, Nr)
            start,
            end,
        )
        iou_score = 1 - (1 - iou_cost) / torch.max(1 - iou_cost) + 1e-2
        
        # instead CLRerNet calculates the iou cost and classification cost
        cost = -iou_score
        return cost
    
    
    def assign(
        self,
        predictions,
        targets,
        img_meta,
    ):
        """
        computes dynamicly matching based on the cost, including cls cost and lane similarity cost
        Args:
            predictions (Dict[torch.Trnsor]): predictions predicted by each stage, including:
                cls_logits: shape (Np, 2), anchor_params: shape (Np, 3),
                lengths: shape (Np, 1) and xs: shape (Np, Nr).
            targets (torch.Tensor): lane targets, shape: (Ng, 6+Nr).
                The first 6 elements are classification targets (2 ch), anchor starting point xy (2 ch),
                anchor theta (1ch) and anchor length (1ch).
            img_meta (dict): meta dict that includes per-image information such as image shape.
        return:
            matched_row_inds (Tensor): matched predictions, shape: (num_targets).
            matched_col_inds (Tensor): matched targets, shape: (num_targets).
        Np: number of priors (anchors), Ng: number of GT lanes, Nr: number of rows.
        """
        img_h, img_w, _ = img_meta.img_shape

        pred_xs = predictions["xs"].detach().clone()  # relative
        target_xs = targets[:, 6:] / (img_w - 1)  # abs -> relative

        # dyanamic k assign by iou
        iou_dynamick = self.iou_dynamick(pred_xs, target_xs)

        if self.cost_combination == 0:  # CLRNet
            cost = self._clrnet_cost(
                predictions, targets, pred_xs, target_xs, img_w, img_h
            )
        elif self.cost_combination == 1:  # CLRerNet
            cost = self._clrernet_cost(predictions, targets, pred_xs, target_xs)
        elif self.cost_combination == 2: # CLRerNet without classification cost
            cost = self._clrernet_cost_wo_cls(predictions, pred_xs, target_xs)
        else:
            raise NotImplementedError(
                f"cost_combination {self.cost_combination} is not implemented!"
            )

        if self.assignment_type == 0: # dynamic_k assignment
            matched_row_inds, matched_col_inds, num_assignments, total_assignments = self.dynamic_k_assign(cost, iou_dynamick)
        elif self.assignment_type == 1: # atss assignment
            matched_row_inds, matched_col_inds = self.atss_assign(cost, iou_dynamick)
        elif self.assignment_type == 2: # atss assignment2 (cost based candidate selection)
            matched_row_inds, matched_col_inds = self.atss_assign2(cost, iou_dynamick)
        else:
            raise NotImplementedError(
                f"assignment_type {self.assignment_type} is not implemented!"
            )

        return matched_row_inds, matched_col_inds, num_assignments, total_assignments
