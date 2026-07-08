from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmaction.registry import MODELS
from mmaction.utils import (ConfigType, SampleList)

@MODELS.register_module()
class AVTV2Head(BaseModule):
    """AVTv2
    ."""
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 clip_len: int = 4,
                 num_classes: int = 4,
                 component: str = 'verb',
                 dataset: str = 'cholect50',
                 center: str = 's',
                 loss_cls: dict = dict(type='CrossEntropyLoss', loss_weight=1.0),
                 init_cfg: Optional[ConfigType] = None,
                ) -> None:
        super().__init__(init_cfg=init_cfg)

        self.in_channels = in_channels
        self.component = component
        self.clip_len = clip_len
        self.dataset = dataset
        self.center = center
        # self.loss_cls = MODELS.build(loss_cls)
        self.fc_cls = nn.Linear(self.in_channels, num_classes)
    
    def get_weights(self, device: torch.device):
        if self.dataset == 'cholect50':
            if self.component == 'verb':
                # wt = [0.52987283, 8.8688099]
                # wt = [0.1380465179681778, 0.03227788582444191, 0.03914199024438858, 0.35290926694869995, 0.5739114284515381, 0.9116821885108948, 
                #      0.5610352158546448, 2.244140863418579, 5.001228332519531, 0.14562644064426422]
                wt = [0.2682666480541229, 0.11874744296073914, 0.1637895703315735, 0.43647709488868713, 1.076643466949463, 1.46815025806427, 
                      0.6163989305496216, 1.46815025806427, 4.249908447265625, 0.13346819579601288]
        elif self.dataset == 'prosta':
            if self.component == 'verb':
                wt = [1.578496832734121, 3.1925207756232687, 2.940051020408163, 0.36857885268838697]
            elif self.component == 'triplet':
                wt = [11.636131593874078, 16.577373737373737, 17.077627471383973, 0.2634920125230794]
        elif self.dataset == 'grasp':
            # wt = [8.188049853372434, 14.442025862068965, 14.59298780487805, 0.26737236063009717]
            # wt = [0.0576963871717453, 0.9808386564254761, 0.8582337498664856, 0.6865870356559753, 0.6058120131492615, 0.9362550377845764, 
            #       1.2873506546020508, 2.0597610473632812, 0.6241700053215027, 1.5844316482543945, 1.3731740713119507, 2.5747013092041016, 
            #       4.1195220947265625, 0.381437212228775, 0.27834609150886536, 0.17604795098304749, 
            #       1.0840846300125122, 0.27463480830192566, 0.15967141091823578, 0.4681275188922882, 0.4291168749332428]
            # adding back cls at the end
            wt = [ 0.40628979,  6.90692641,  6.04356061,  4.83484848,  4.26604278,
                6.59297521,  9.06534091, 14.50454545,  4.3953168 , 11.15734266,
                9.66969697, 18.13068182, 29.00909091,  2.68602694,  1.96007371,
                1.25039185,  7.63397129,  1.93393939,  1.12438337,  3.2964876 ,
                3.0217803 ,  0.07096157]
        elif self.dataset == 'mbp':
            if self.center == 's':
                wt = [16.386742756804214, 38.29924760601915, 38.53647625602202, 0.2572734123009346]
            elif self.center == 'ac':
                wt = [17.06734693877551, 34.94712936046512, 35.17721287490856, 0.257442240823177]
            elif self.center == 'b':
                wt = [18.115870153291254, 31.148062015503875, 31.36690085870414, 0.2576778786032642]
            elif self.center == 's1':
                wt = [16.43384929906542, 38.30462899931926, 38.30462899931926, 0.25727198741747287]
            elif self.center == 'ac1':
                wt = [17.13809271054494, 34.906126126126125, 34.93130183916336, 0.25744170223623153]
            elif self.center == 'b1':
                wt = [18.220377019748653, 31.083460949464012, 31.131134969325153, 0.25767731779461983]
        return torch.tensor(wt).to(device)
    
    def get_labels(self, data_samples: SampleList) -> torch.Tensor:
        if self.component == 'verb':
            labels_list = [torch.as_tensor(sample.sc_v) for sample in data_samples]
            # Multi-class case: each sample provides scalar / [1] class id.
            if labels_list[0].ndim == 0:
                labels = torch.stack(labels_list, dim=0).long()  # [B]
            else:
                labels = torch.stack([x.reshape(-1) for x in labels_list], dim=0)  # [B, K]
                if labels.shape[1] == 1:
                    labels = labels.squeeze(1).long()  # [B]
                else:
                    labels = labels.float()  # [B, C] multi-label
            # labels = torch.cat([sample.cl_v for sample in data_samples]).float()
        elif self.component == 'step':
            labels = torch.cat([torch.as_tensor(sample.sc_v) for sample in data_samples])
        elif self.component == 'triplet':
            labels = torch.cat([sample.sc_ivt for sample in data_samples])
        elif self.component == 'instrument-target':
            labels = torch.cat([sample.sc_it for sample in data_samples])
        elif self.component == 'instrument-verb':
            labels = torch.cat([sample.sc_iv for sample in data_samples])
        elif self.component == 'instrument':
            labels = torch.cat([sample.sc_i for sample in data_samples])
        return labels

    def forward(self, inputs: torch.Tensor, **kwargs) -> dict:
        return self.fc_cls(inputs)  # [B, T, C]

    def loss(self, 
            inputs: torch.Tensor, 
            data_samples: SampleList,
            **kwargs) -> dict:
        logits = self(inputs)
        losses = dict()
        labels = self.get_labels(data_samples).to(inputs.device)

        if self.component == 'verb':
            class_weight = self.get_weights(inputs.device).float()
            if labels.ndim == 1:
                # hardcoded multiclass CE
                loss_cls = torch.nn.functional.cross_entropy(logits, labels.long(), weight=class_weight)
            else:
                # hardcoded multiclass CE from [B,1] ids or one-hot [B,C]
                if labels.shape[-1] == 1:
                    labels_mc = labels.squeeze(-1).long()
                else:
                    labels_mc = labels.argmax(dim=-1).long()
                loss_cls = torch.nn.functional.cross_entropy(logits, labels_mc, weight=class_weight)
        elif self.component == 'step':
            class_weight = self.get_weights(inputs.device).float()
            loss_cls = torch.nn.functional.cross_entropy(
                logits, labels.long().reshape(-1), weight=class_weight)
            
        losses[f'loss_cls_{self.component}'] = loss_cls
        return losses

    def predict(self, 
                inputs: torch.Tensor, 
                data_samples: SampleList,
                **kwargs) -> SampleList:
        
        logits = self(inputs)
        if self.component == 'verb':
            labels = self.get_labels(data_samples).to(inputs.device)
            if labels.ndim == 1:
                scores = logits.softmax(dim=-1)
            else:
                scores = logits.sigmoid()
        else:
            scores = logits.softmax(dim=-1)
            labels = self.get_labels(data_samples).to(inputs.device)

        for data_sample, score, label in zip(data_samples, scores, labels):
            data_sample.set_pred_score(score)
            data_sample.set_pred_label(label)

        return data_samples
