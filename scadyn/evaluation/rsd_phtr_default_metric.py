
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
    default_prefix: Optional[str] = 'rsd_phase_tr'

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
        elif self.task == 'phase-transition':
            # logits shape b x (3 x 7) , 3 time slots 2, 3, 5min
            phase_preds = preds
            phase_labels = labels

            # split into 3 time slots
            p_2min = phase_preds[:, :7]
            p_3min = phase_preds[:, 7:14]
            p_5min = phase_preds[:, 14:]

            l_2min = phase_labels[:, :7]
            l_3min = phase_labels[:, 7:14]
            l_5min = phase_labels[:, 14:]

            per_slot_mae = []
            # eval each time slot and save results and average over all slots
            for p, l in zip([p_2min, p_3min, p_5min], [l_2min, l_3min, l_5min]):
                mae = mean_absolute_error(p.contiguous(), l.contiguous()) # NOTE: contiguous to avoid shape mismatch error
                mae = round(to_np(mae), 4)
                per_slot_mae.append(mae)

            mae = np.mean(per_slot_mae)
            mae = round(mae, 4)
            log_results['mae_phase_transition'] = mae
            log_results['mae_phase_transition_2min'] = per_slot_mae[0]
            log_results['mae_phase_transition_3min'] = per_slot_mae[1]
            log_results['mae_phase_transition_5min'] = per_slot_mae[2]

            # mae = mean_absolute_error(phase_preds, phase_labels)
            # mae = round(to_np(mae), 4)
            # log_results['mae_phase_transition'] = mae

            # TODO: MAE_e, MAE_p
            logger.info(colored(f"{self.task} MAE: mean: {mae} | 2min: {per_slot_mae[0]} | 3min: {per_slot_mae[1]} | 5min: {per_slot_mae[2]}", 'green'))
        elif self.task == 'step-transition':
            step_preds = preds
            step_labels = labels
            mae = mean_absolute_error(step_preds.squeeze(-1), step_labels)
            mae = round(to_np(mae), 4)
            log_results['mae_step_transition'] = mae
            logger.info(colored(f"MAE {self.task}: {mae}", 'green'))

        return log_results