from typing import List
from typing import Union

import torch
from scipy.optimize import linear_sum_assignment

from nora.structures import Instances
from .base import Matcher
from .match_cost import MatchCost

__all__ = ["HungarianMatcher"]


class HungarianMatcher(Matcher):
    def __init__(self, match_costs: Union[List[MatchCost], MatchCost]):
        super().__init__()

        if isinstance(match_costs, MatchCost):
            match_costs = [match_costs]

        self.match_costs = match_costs

    def match(self, pred_instances: Instances, gt_instances: Instances):
        num_gt = len(gt_instances)
        num_pred = len(pred_instances)

        if num_gt == 0 or num_pred == 0:
            default_matches = gt_instances.gt_classes.new_full((num_pred,), 0, dtype=torch.int64)
            default_match_labels = gt_instances.gt_classes.new_full((num_pred,), 0, dtype=torch.int8)
            return default_matches, default_match_labels

        costs = []
        for match_cost in self.match_costs:
            cost = match_cost(pred_instances, gt_instances)
            costs.append(cost)

        costs = torch.stack(costs).sum(dim=0)

        costs = costs.detach().cpu()
        matched_row_inds, matched_col_inds = linear_sum_assignment(costs)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(gt_instances.gt_classes.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(gt_instances.gt_classes.device)

        match_labels = gt_instances.gt_classes.new_full((num_pred,), 0, dtype=torch.int8)
        match_labels[matched_row_inds] = 1
        matches = gt_instances.gt_classes.new_full((num_pred,), 0, dtype=torch.int64)
        matches[matched_row_inds] = matched_col_inds
        return matches, match_labels
