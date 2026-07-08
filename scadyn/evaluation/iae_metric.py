
import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple

import mmengine
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.evaluator import BaseMetric
from terminaltables import AsciiTable
from termcolor import colored

from mmaction.registry import METRICS
from mmengine.logging import MMLogger

from torchmetrics.functional.classification import (multiclass_f1_score, multiclass_precision, 
                                                    multiclass_recall, multiclass_accuracy,
                                                    multiclass_roc,
                                                    multiclass_auroc,
                                                    multiclass_average_precision,
                                                    multiclass_confusion_matrix)

def to_tensor(value):
    """Convert value to torch.Tensor."""
    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value)
    elif isinstance(value, Sequence) and not mmengine.is_str(value):
        value = torch.tensor(value)
    elif not isinstance(value, torch.Tensor):
        raise TypeError(f'{type(value)} is not an available argument.')
    return value

to_np = lambda x: x.detach().cpu().numpy().item()

@METRICS.register_module()
class IAE_Metric(BaseMetric):
    """Metric to evaluate Intraoperative Adverse Event (IAE) anticipation at two levels - presence and grade."""
    default_prefix: Optional[str] = 'iae'
    def __init__(self,
                 metric_list: List[str] = [],
                 average:str ='macro',
                 collect_device: str = 'cpu',
                 num_classes: int = 9,
                 task: str = 'rsd',
                 multi_label: bool = False,
                 ignore_windows: List[int] = [7],
                 metric_options: Optional[Dict] = dict(
                     top_k_accuracy=dict(topk=(1, ))),
                 prefix: Optional[str] = None) -> None:

        super().__init__(collect_device=collect_device, prefix=prefix)

        self.average = average
        self.num_classes = num_classes
        self.task = task
        self.multi_label = multi_label
        self.metric_list = metric_list
        self.ignore_windows = ignore_windows

    def process(self, data_batch: Sequence[Tuple[Any, Dict]],
                data_samples: Sequence[Dict]) -> None:
        data_samples = copy.deepcopy(data_samples)
        for data_sample in data_samples:
            preds = data_sample['pred_score'].detach().cpu()
            gts   = data_sample['pred_label'].detach().cpu()
            window = data_sample['anticipation_window'].detach().cpu()
            self.results.append({'gt':gts, 'pred':preds, 'window':window})

    def compute_metrics(self, results: List) -> Dict:
        logger: MMLogger = MMLogger.get_current_instance()
        logger.info(f"Starting {self.task} evaluation")

        log_results = dict()
        preds  = torch.stack([result['pred'] for result in results], dim=0)
        labels = torch.cat([result['gt'] for result in results], dim=0)
        windows = torch.cat([result['window'] for result in results], dim=0)

        unique_windows = torch.unique(windows)
        per_window_results = {window.item(): {} for window in unique_windows if window.item() not in self.ignore_windows}
        per_window_classwise_results = {window.item(): {} for window in unique_windows if window.item() not in self.ignore_windows}

        for window in unique_windows:
            window_item = window.item()
            if window_item in self.ignore_windows:
                continue  # Skip ignored windows

            mask = windows == window
            preds_window = preds[mask]
            labels_window = labels[mask]

            clsAP = multiclass_average_precision(preds_window, labels_window, num_classes=self.num_classes, average=None)
            mAP = clsAP.mean()
            mAP = round(mAP.item(), 4)
            per_window_results[window_item]['mAP'] = mAP
            per_window_classwise_results[window_item]['AP'] = clsAP.tolist()

            clsf1 = multiclass_f1_score(preds_window, labels_window, num_classes=self.num_classes, average=None)
            f1 = clsf1.mean()
            f1 = round(f1.item(), 4)
            per_window_results[window_item]['f1'] = f1
            per_window_classwise_results[window_item]['f1'] = clsf1.tolist()

            clsAUROC = multiclass_auroc(preds_window, labels_window, num_classes=self.num_classes, average=None)
            AUROC = clsAUROC.mean()
            AUROC = round(AUROC.item(), 4)
            per_window_results[window_item]['AUROC'] = AUROC
            per_window_classwise_results[window_item]['AUROC'] = clsAUROC.tolist()

            clsAccuracy = multiclass_accuracy(preds_window, labels_window, num_classes=self.num_classes, average=None)
            Accuracy = clsAccuracy.mean()
            Accuracy = round(Accuracy.item(), 4)
            per_window_results[window_item]['Accuracy'] = Accuracy
            per_window_classwise_results[window_item]['Accuracy'] = clsAccuracy.tolist()

            clsPrecision = multiclass_precision(preds_window, labels_window, num_classes=self.num_classes, average=None)
            Precision = clsPrecision.mean()
            Precision = round(Precision.item(), 4)
            per_window_results[window_item]['Precision'] = Precision
            per_window_classwise_results[window_item]['Precision'] = clsPrecision.tolist()

            clsRecall = multiclass_recall(preds_window, labels_window, num_classes=self.num_classes, average=None)
            Recall = clsRecall.mean()
            Recall = round(Recall.item(), 4)
            per_window_results[window_item]['Recall'] = Recall
            per_window_classwise_results[window_item]['Recall'] = clsRecall.tolist()

        # Calculate overall metrics
        overall_mAP = np.mean([results['mAP'] for results in per_window_results.values()])
        overall_f1 = np.mean([results['f1'] for results in per_window_results.values()])
        overall_AUROC = np.mean([results['AUROC'] for results in per_window_results.values()])
        overall_Accuracy = np.mean([results['Accuracy'] for results in per_window_results.values()])
        overall_Precision = np.mean([results['Precision'] for results in per_window_results.values()])
        overall_Recall = np.mean([results['Recall'] for results in per_window_results.values()])

        logger.info(colored(f"Overall IAE Anticipation mAP: {overall_mAP:.4f}", 'green'))
        logger.info(colored(f"Overall IAE Anticipation F1: {overall_f1:.4f}", 'green'))
        logger.info(colored(f"Overall IAE Anticipation AUROC: {overall_AUROC:.4f}", 'green'))
        logger.info(colored(f"Overall IAE Anticipation Accuracy: {overall_Accuracy:.4f}", 'green'))
        logger.info(colored(f"Overall IAE Anticipation Precision: {overall_Precision:.4f}", 'green'))
        logger.info(colored(f"Overall IAE Anticipation Recall: {overall_Recall:.4f}", 'green'))

        log_results['overall_mAP'] = overall_mAP
        log_results['overall_f1'] = overall_f1
        log_results['overall_AUROC'] = overall_AUROC
        log_results['overall_Accuracy'] = overall_Accuracy
        log_results['overall_Precision'] = overall_Precision
        log_results['overall_Recall'] = overall_Recall
        log_results['per_window_results'] = per_window_results
        # log_results['per_window_classwise_results'] = per_window_classwise_results

        # Print per window classwise results per window and each metric should be the header
        for window in unique_windows:
            window_item = window.item()
            if window_item in self.ignore_windows:
                continue  # Skip ignored windows

            logger.info(colored(f"Window: {window_item}", 'blue'))
            for metric in per_window_classwise_results[window_item]:
                # round the metric to 4 decimal places but its a list
                metric_value = [round(value, 4) for value in per_window_classwise_results[window_item][metric]]
                logger.info(colored(f"{metric}: {metric_value}", 'red'))

        return log_results