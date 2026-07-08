
import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import mmengine
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.evaluator import BaseMetric
from terminaltables import AsciiTable
from termcolor import colored

from mmaction.registry import METRICS
from torchmetrics.functional.classification import (multiclass_f1_score, multiclass_precision, 
                                                    multiclass_recall, multiclass_accuracy,
                                                    multiclass_average_precision,
                                                    multilabel_average_precision,
                                                    multilabel_precision,
                                                    multilabel_accuracy,
                                                    multilabel_recall,
                                                    multilabel_f1_score)
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
class FeaturePredictionAndClassificationMetric(BaseMetric):
    """Evaluation Metric for Classification and feature generation."""
    default_prefix: Optional[str] = 'mAP'
    def __init__(self,
                 metric_list: List[str] = [],
                 average:str ='macro',
                 collect_device: str = 'cpu',
                 num_states: int = 4,
                 num_classes: int = 90,
                 multi_label: bool = False,
                 metric_options: Optional[Dict] = dict(
                     top_k_accuracy=dict(topk=(1, ))),
                 prefix: Optional[str] = None) -> None:

        super().__init__(collect_device=collect_device, prefix=prefix)

        self.average = average
        self.num_states = num_states
        self.num_classes = num_classes
        self.feats = []

    def process(self, data_batch: Sequence[Tuple[Any, Dict]],
                data_samples: Sequence[Dict]) -> None:
        data_samples = copy.deepcopy(data_samples)
        data_samples, feats = data_samples
        for data_sample in data_samples:
            preds, gts = dict(), dict()
            preds['logits'] = data_sample.pred_score.detach().cpu()
            gts['labels']   = data_sample.pred_label.detach().cpu()

            # future clip predicted logits and class label
            preds['pred_futr_logits'] = data_sample.pred_future_logits.detach().cpu()
            preds['gt_futr_logits']   = data_sample.gt_future_logits.detach().cpu()
            gts['fc_labels']          = data_sample.fc_labels.detach().cpu()
            self.results.append({'gt':gts, 'pred':preds})

        # collect gt features and predicted features (future)
        self.feats.append({'gt':feats['gt_future'].cpu(), 
                           'pred':feats['pred_future'].cpu()})

    def compute_metrics(self, results: List) -> Dict:
        logger: MMLogger = MMLogger.get_current_instance()
        logger.info(f"Starting evaluation for Number of states: {self.num_states}")

        log_results = dict()

        # classification results, mean AP on current clip features
        preds  = torch.stack([result['pred']['logits'] for result in results], dim=0) # b x num_classes
        labels = torch.stack([result['gt']['labels'] for result in results], dim=0).long() # b x num_classes
        cls_AP, mAP = self.get_results(preds, labels, num_classes=self.num_classes, metric='mAP')
        log_results['mAP'] = mAP
        
        colored_mAP = colored(f'mAP: {mAP}', 'green')
        colored_cls_AP = colored(f'cls_AP: {cls_AP}', 'blue')
        logger.info(f"Mean Average Precision: {colored_mAP}")
        logger.info(f"Class Average Precision: {colored_cls_AP}")

        # future clip classification results, mean AP
        labels = torch.stack([result['gt']['fc_labels'] for result in results], dim=0).long()
        futr_pred_logits = torch.stack([result['pred']['pred_futr_logits'] for result in results], dim=0)
        futr_gt_logits   = torch.stack([result['pred']['gt_futr_logits'] for result in results], dim=0)
        
        fp_cls_AP, mAP = self.get_results(futr_pred_logits, labels, num_classes=self.num_classes, metric='mAP')
        colored_fp_AP  = colored(f'future_pred_mAP: {mAP}', 'green')
        log_results['future_pred_mAP'] = mAP
        fg_cls_AP, mAP = self.get_results(futr_gt_logits, labels, num_classes=self.num_classes, metric='mAP')
        colored_fg_AP  = colored(f'future_gt_mAP: {mAP}', 'green')
        logger.info(colored_fp_AP)
        logger.info(colored_fg_AP)
        log_results['future_gt_mAP'] = mAP

        colored_fp_AP = colored(f'future_pred_mAP: {fp_cls_AP}', 'green')
        colored_fg_AP = colored(f'future_gt_mAP: {fg_cls_AP}', 'green')
        logger.info(colored_fp_AP)
        logger.info(colored_fg_AP)
        # log_results['future_pred_cls_AP'] = fp_cls_AP
        # log_results['future_gt_cls_AP'] = fg_cls_AP

        # classwise AP on future clip classification (both predicted and gt)
        # NOTE: gt for future means the model takes future clips as input compared to
        # predicted where the previous clip is input to model.
        colored_fp_cls_AP = colored(f'future_pred_cls_AP: {fp_cls_AP}', 'blue')
        colored_fg_cls_AP = colored(f'future_gt_cls_AP: {fg_cls_AP}', 'blue')
        logger.info(colored_fp_cls_AP)
        logger.info(colored_fg_cls_AP)

        # feature similarity results
        pred_future = torch.cat([result['pred'] for result in self.feats], dim=0)
        gt_future   = torch.cat([result['gt'] for result in self.feats], dim=0)
        feature_sim = F.cosine_similarity(pred_future, gt_future, dim=2).mean()
        log_results['feature_similarity'] = feature_sim

        return log_results

    def get_results(self, pred, label, num_classes=4, metric='mAP'):
        if metric == 'mAP':
            clsAP = multilabel_average_precision(pred, label, num_classes, average='none')
            mAP = torch.nanmean(clsAP) # ignoring background
            mAP = round(mAP.detach().cpu().numpy().item(), 4)
        
        return clsAP, mAP