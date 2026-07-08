
import copy
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import pdb
import mmengine
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.evaluator import BaseMetric
from terminaltables import AsciiTable
from termcolor import colored

from mmaction.registry import METRICS
from torchmetrics.functional import mean_absolute_error
from mmengine.logging import MMLogger

def to_tensor(value):
    """Convert value to torch.Tensor."""
    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value)
    elif isinstance(value, Sequence) and not mmengine.is_str(value):
        value = torch.tensor(value)
    elif not isinstance(value, torch.Tensor):
        raise TypeError(f'{type(value)} is not an available argument.')
    return value

to_np = lambda x: x.detach().cpu().item()

@METRICS.register_module()
class RSDPhaseTransitionMetric(BaseMetric):
    """Metric to evaluate Remaining surgery duration & phase transition prediction."""
    default_prefix: Optional[str] = 'long_term_anticipation'

    def __init__(self,
                 metric_list: List[str] = [],
                 average:str ='macro',
                 collect_device: str = 'cpu',
                 num_classes: int = 90,
                 task: str = 'rsd',
                 multi_label: bool = False,
                 event_time_markers: list = [7, 14],
                 metric_options: Optional[Dict] = dict(
                     top_k_accuracy=dict(topk=(1, ))),
                 prefix: Optional[str] = None) -> None:

        super().__init__(collect_device=collect_device, prefix=prefix)

        self.average = average
        self.num_classes = num_classes
        self.event_time_markers = event_time_markers
        self.task = task
        self.multi_label = multi_label
        self.metric_list = metric_list

    def process(self, data_batch: Sequence[Tuple[Any, Dict]],
                data_samples: Sequence[Dict]) -> None:
        data_samples = copy.deepcopy(data_samples)
        for data_sample in data_samples:
            preds = data_sample['pred_score'].detach().cpu()
            gts   = data_sample['pred_label'].detach().cpu()
            self.results.append({'gt':gts, 'pred':preds})

    def compute_metrics(self, results: List) -> Dict:
        logger: MMLogger = MMLogger.get_current_instance()
        logger.info(f"Starting {self.task} evaluation")

        log_results = dict()
        preds  = torch.stack([result['pred'] for result in results], dim=0)
        labels = torch.cat([result['gt'] for result in results], dim=0)

        if self.task == 'rsd':
            rsd_preds = preds
            rsd_labels = labels
            mae = mean_absolute_error(rsd_preds.squeeze(-1), rsd_labels)
            mae = round(to_np(mae), 4)
            log_results['mae_rsd'] = mae
            logger.info(colored(f"MAE {self.task}: {mae}", 'green'))
        elif 'transition' in self.task:
            # logits shape b x (3 x C) , 3 time slots 2, 3, 5min
            # C is 7 for cholect50, 11 for phase in grasp, ? for steps in grasp

            # split into 3 time slots
            t1, t2 = self.event_time_markers
            p_2min = preds[:, :t1]
            p_3min = preds[:, t1:t2]
            p_5min = preds[:, t2:]
            l_2min = labels[:, :t1]
            l_3min = labels[:, t1:t2]
            l_5min = labels[:, t2:]

            per_slot_mae = []
            # eval each time slot and save results and average over all slots
            for p, l in zip([p_2min, p_3min, p_5min], [l_2min, l_3min, l_5min]):
                mae = mean_absolute_error(p.contiguous(), l.contiguous()) # NOTE: contiguous to avoid shape mismatch error
                mae = round(to_np(mae), 4)
                per_slot_mae.append(mae)

            mae = np.mean(per_slot_mae)
            mae = round(mae, 4)
            log_results[f'mae_{self.task}'] = mae
            log_results[f'mae_{self.task}_2min'] = per_slot_mae[0]
            log_results[f'mae_{self.task}_3min'] = per_slot_mae[1]
            log_results[f'mae_{self.task}_5min'] = per_slot_mae[2]
            logger.info(colored(f"{self.task} MAE: mean: {mae} | 2min: {per_slot_mae[0]} | 3min: {per_slot_mae[1]} | 5min: {per_slot_mae[2]}", 'green'))
        elif 'anticipation' in self.task:
            # logits shape b x (3 x C) , 3 time slots 2, 3, 5min
            # split into 3 time slots
            t1, t2 = self.event_time_markers
            p_1min = preds[:, :t1]
            p_2min = preds[:, t1:t2]
            p_3min = preds[:, t2:]
            l_1min = labels[:, :t1]
            l_2min = labels[:, t1:t2]
            l_3min = labels[:, t2:]

            per_slot_mae = []
            # eval each time slot and save results and average over all slots
            for p, l in zip([p_1min, p_2min, p_3min], [l_1min, l_2min, l_3min]):
                mae = mean_absolute_error(p.contiguous(), l.contiguous()) # NOTE: contiguous to avoid shape mismatch error
                mae = round(to_np(mae), 4)
                per_slot_mae.append(mae)

            mae = np.mean(per_slot_mae)
            mae = round(mae, 4)
            log_results[f'mae_{self.task}'] = mae
            log_results[f'mae_{self.task}_1min'] = per_slot_mae[0]
            log_results[f'mae_{self.task}_2min'] = per_slot_mae[1]
            log_results[f'mae_{self.task}_3min'] = per_slot_mae[2]
            # logger.info(colored(f"{self.task} MAE: mean: {mae} | 1min: {per_slot_mae[0]} | 2min: {per_slot_mae[1]} | 3min: {per_slot_mae[2]}", 'green'))
            logger.info(colored(f"{self.task} MAE: mean: {mae} | 2min: {per_slot_mae[0]} | 3min: {per_slot_mae[1]} | 5min: {per_slot_mae[2]}", 'green'))

        return log_results