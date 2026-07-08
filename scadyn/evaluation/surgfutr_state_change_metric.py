
import copy
from collections import OrderedDict
from itertools import product
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
                                                    multiclass_auroc,
                                                    multiclass_average_precision,
                                                    multiclass_confusion_matrix,
                                                    multilabel_average_precision)
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
class SurgFutrStateChangeMetric(BaseMetric):
    """State Change Metric based on either current clip features or 
       predicted future clip features.
    ."""
    default_prefix: Optional[str] = 'mAP'

    def __init__(self,
                 metric_list: List[str] = [],
                 average:str ='macro',
                 collect_device: str = 'cpu',
                 num_states: int = 4,
                 multi_label: bool = False,
                 prefix: Optional[str] = None) -> None:

        super().__init__(collect_device=collect_device, prefix=prefix)

        self.average = average
        self.num_states = num_states
        self.multi_label = multi_label
        self.metric_list = metric_list
        
        self.verb_results = []
        self.target_results = []
        self.feats = []

    def process(self, data_batch: Sequence[Tuple[Any, Dict]],
                data_samples: Sequence[Dict]) -> None:
        data_samples = copy.deepcopy(data_samples)
        data_samples, feats = data_samples
        for data_sample in data_samples:
            preds = data_sample.pred_score.detach().cpu()
            gts   = data_sample.pred_label.detach().cpu()
            self.results.append({'gt':gts, 'pred':preds})

        self.feats.append({'gt':feats['gt_future'].cpu(), 
                           'pred':feats['pred_future'].cpu()})

    def compute_metrics(self, results: List) -> Dict:
        logger: MMLogger = MMLogger.get_current_instance()
        logger.info(f"Starting evaluation for Number of states: {self.num_states}")

        log_results = dict()
        preds  = torch.cat([result['pred'] for result in results], dim=0)  # b x num_classes x num_states
        labels = torch.cat([result['gt'] for result in results], dim=0)    # b x num_classes

        # 1. compute state change metric > over all samples or over per class and then average across class
        cls_AP, mAP = self.get_results(preds, labels, num_classes=self.num_states, metric='mAP')
        cls_AP = [round(to_np(p), 4) for p in cls_AP]
        ct, dt, ot, bg = cls_AP
        logger.info(colored(f"mAP {mAP} - Statewise AP: Continuity: {ct}, Discontinuity: {dt}, Onset: {ot}, Background: {bg}", 'green'))
        log_results['state_mAP'] = mAP
        log_results['continuity_mAP'] = ct
        log_results['discontinuity_mAP'] = dt
        log_results['onset_mAP'] = ot

        # auroc per state
        cls_auroc, auroc = self.get_results(preds, labels, num_classes=self.num_states, metric='auroc')
        cls_auroc = [round(to_np(p), 4) for p in cls_auroc]
        ct_auroc, dt_auroc, ot_auroc, bg_auroc = cls_auroc
        logger.info(colored(f"mAUROC {auroc} - Statewise AUROC: Continuity: {ct_auroc}, Discontinuity: {dt_auroc}, Onset: {ot_auroc}, Background: {bg_auroc}", 'green'))

        # f1 per state
        cls_f1, f1 = self.get_results(preds, labels, num_classes=self.num_states, metric='f1')
        cls_f1 = [round(to_np(p), 4) for p in cls_f1]
        ct_f1, dt_f1, ot_f1, bg_f1 = cls_f1
        logger.info(colored(f"F1 {f1} - Statewise F1: Continuity: {ct_f1}, Discontinuity: {dt_f1}, Onset: {ot_f1}, Background: {bg_f1}", 'green'))

        # precision 
        cls_prec, prec = self.get_results(preds, labels, num_classes=self.num_states, metric='prec')
        cls_prec = [round(to_np(p), 4) for p in cls_prec]
        ct_prec, dt_prec, ot_prec, bg_prec = cls_prec
        logger.info(colored(f"Precision {prec} - Statewise Precision: Continuity: {ct_prec}, Discontinuity: {dt_prec}, Onset: {ot_prec}, Background: {bg_prec}", 'blue'))

        # recall
        cls_recall, recall = self.get_results(preds, labels, num_classes=self.num_states, metric='recall')
        cls_recall = [round(to_np(p), 4) for p in cls_recall]
        ct_recall, dt_recall, ot_recall, bg_recall = cls_recall
        logger.info(colored(f"Recall {recall} - Statewise Recall: Continuity: {ct_recall}, Discontinuity: {dt_recall}, Onset: {ot_recall}, Background: {bg_recall}", 'green'))

        # conf matrix
        conf_matrix = self.get_results(preds, labels, num_classes=self.num_states, metric='conf_matrix')
        logger.info(colored(f"Confusion Matrix: {conf_matrix}", 'blue'))

        # feature similarity
        pred_future = torch.cat([result['pred'] for result in self.feats], dim=0)
        gt_future   = torch.cat([result['gt'] for result in self.feats], dim=0)
        feature_sim = F.cosine_similarity(pred_future, gt_future, dim=2).mean()
        log_results['feature_similarity'] = feature_sim

        return log_results

    def get_results(self, pred, label, num_classes=4, metric='mAP'):
        # print("label in metric", label.shape)
        if metric == 'mAP':
            if self.multi_label:
                clsAP = multilabel_average_precision(pred, label.long(), num_classes, average='none')
            else:
                clsAP = multiclass_average_precision(pred, label, num_classes, average='none')
            mAP = torch.nanmean(clsAP[:-1]) # ignoring background
            mAP = round(mAP.detach().cpu().numpy().item(), 4)
            return clsAP, mAP
        elif metric == 'f1':
            clsF1 = multiclass_f1_score(pred, label, num_classes, average='none')
            clsF1mean = torch.nanmean(clsF1[:-1])
            clsF1mean = round(clsF1mean.detach().cpu().numpy().item(), 4)
            return clsF1, clsF1mean
        elif metric == 'prec':
            clsPrec = multiclass_precision(pred, label, num_classes, average='none')
            clsPrecmean = torch.nanmean(clsPrec[:-1])
            clsPrecmean = round(clsPrecmean.detach().cpu().numpy().item(), 4)
            return clsPrec, clsPrecmean
        elif metric == 'recall':
            clsRecall = multiclass_recall(pred, label, num_classes, average='none')
            clsRecallmean = torch.nanmean(clsRecall[:-1])
            clsRecallmean = round(clsRecallmean.detach().cpu().numpy().item(), 4)
            return clsRecall, clsRecallmean
        elif metric == 'auroc':
            auroc = multiclass_auroc(pred, label, num_classes, average='none')
            aurocmean = torch.nanmean(auroc[:-1])
            aurocmean = round(aurocmean.detach().cpu().numpy().item(), 4)
            return auroc, aurocmean
        elif metric == 'conf_matrix':
            conf_matrix = multiclass_confusion_matrix(pred, label, num_classes)
            return conf_matrix