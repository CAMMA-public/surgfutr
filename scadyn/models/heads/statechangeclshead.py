from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmaction.registry import MODELS
from mmaction.utils import (ConfigType, SampleList)

@MODELS.register_module()
class StateChangeClassHead(BaseModule):
    """State Change Head to predict state change from the class embeddings."""
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 clip_len: int = 4,
                 num_states: int = 4,
                 component: str = 'verb',
                 dataset: str = 'cholect50',
                 center: str = 's',
                 stability_duration: int = 3,
                 loss_state: dict = dict(type='CrossEntropyLoss', loss_weight=1.0),
                 init_cfg: Optional[ConfigType] = None,
                ) -> None:
        super().__init__(init_cfg=init_cfg)

        self.in_channels = in_channels
        self.component = component
        self.clip_len = clip_len
        self.dataset = dataset
        self.center = center
        self.stability_duration = stability_duration

        self.loss_cls = MODELS.build(loss_state)
        self.fc_state = nn.Linear(self.in_channels, num_states)

    def forward(self, inputs: torch.Tensor, **kwargs) -> dict:
        return self.fc_state(inputs)
    
    def get_weights(self, device: torch.device):
        if self.dataset == 'cholect50':
            if self.component == 'verb':
                wt = [1.9524793388429753, 9.218555417185554, 8.544632550981147, 0.30653020097173145]
                if self.stability_duration == 1:
                    wt = [1.95069935, 8.41315877, 7.78365532, 0.30863935] # NOTE: for annotation where stability duration is set to 1s not 3s the default.
                elif self.stability_duration == 5:
                    wt = [1.90327793, 10.08439335, 9.24327354, 0.30606866]
            elif self.component == 'triplet':
                wt = [18.72154779969651, 77.67576075550892, 73.02696481420585, 0.25510086865823206]
            elif self.component == 'instrument':
                wt = [1.1214965070280278, 10.14041095890411, 9.409957627118644, 0.3444180215576292]
            elif self.component == 'instrument-verb':
                wt = [5.06264796142043, 22.5544921875, 21.095907928388748, 0.26948836905384216]
            elif self.component == 'instrument-target':
                wt = [11.01784813319879, 49.762343334599315, 46.64444642221431, 0.25855133464755714]
        elif self.dataset == 'prosta':
            if self.component == 'verb':
                wt = [1.578496832734121, 3.1925207756232687, 2.940051020408163, 0.36857885268838697]
            elif self.component == 'triplet':
                wt = [11.636131593874078, 16.577373737373737, 17.077627471383973, 0.2634920125230794]
        elif self.dataset == 'grasp':
            wt = [8.188049853372434, 14.442025862068965, 14.59298780487805, 0.26737236063009717]
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
            labels = torch.cat([sample.sc_v for sample in data_samples])
        elif self.component == 'triplet':
            labels = torch.cat([sample.sc_ivt for sample in data_samples])
        elif self.component == 'instrument-target':
            labels = torch.cat([sample.sc_it for sample in data_samples])
        elif self.component == 'instrument-verb':
            labels = torch.cat([sample.sc_iv for sample in data_samples])
        elif self.component == 'instrument':
            labels = torch.cat([sample.sc_i for sample in data_samples])
        return labels

    def loss(self, 
            inputs: torch.Tensor, 
            data_samples: SampleList,
            **kwargs) -> dict:
        state_logits = self(inputs)
        losses = dict()
        labels = self.get_labels(data_samples)
        class_weight = self.get_weights(inputs.device)
        loss_sc = self.loss_cls(state_logits.permute(0, 2, 1), labels, weight=class_weight)
        losses[f'loss_sc_{self.component}'] = loss_sc

        losses['state_logits'] = state_logits

        return losses

    def predict(self, 
                inputs: torch.Tensor, 
                data_samples: SampleList,
                **kwargs) -> SampleList:
        
        state_logits = self(inputs)
        state_logits = state_logits.softmax(dim=-1)
        labels = self.get_labels(data_samples)

        for data_sample, logit, label in zip(data_samples, state_logits, labels):
            data_sample.set_pred_score(logit)
            data_sample.set_pred_label(label)

        return data_samples

@MODELS.register_module()
class StateChangeClassHeadV2(BaseModule):
    """State Change Head to predict state change from the class embeddings."""
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 clip_len: int = 4,
                 num_states: int = 4,
                 component: str = 'verb',
                 dataset: str = 'cholect50',
                 center: str = 's',
                 loss_state: dict = dict(type='CrossEntropyLoss', loss_weight=1.0),
                 init_cfg: Optional[ConfigType] = None,
                ) -> None:
        super().__init__(init_cfg=init_cfg)

        self.in_channels = in_channels
        self.component = component
        self.clip_len = clip_len
        self.dataset = dataset
        self.center = center
        self.loss_cls = MODELS.build(loss_state)
        self.fc_state = nn.Linear(self.in_channels, num_states)

    def forward(self, inputs: torch.Tensor, **kwargs) -> dict:
        return self.fc_state(inputs)
    
    def get_weights(self, device: torch.device):
        if self.dataset == 'cholect50':
            if self.component == 'verb':
                # wt = [1.9524793388429753, 9.218555417185554, 8.544632550981147, 0.30653020097173145]
                wt = [
                    [0.4667674 , 1.8912448 , 1.57708454, 0.06490326],
                    [0.23218334, 1.73520771, 1.63313667, 0.39947228],
                    [0.21339266, 1.83444416, 1.73397764, 0.21818555],
                    [0.84753742, 1.62527764, 1.49349837, 0.03368656],
                    [0.52668229, 1.80140609, 1.6572936 , 0.01461802],
                    [0.657827  , 1.68111344, 1.65054774, 0.01051183],
                    [0.97134567, 1.57537575, 1.43106652, 0.02221205],
                    [2.33815624, 0.80626077, 0.85023863, 0.00534436],
                    [1.2540553 , 1.48919067, 1.2540553 , 0.00269873],
                    [1.22519145, 1.3767169 , 1.28796986, 0.11012179]
                    ]
            elif self.component == 'triplet':
                wt = [18.72154779969651, 77.67576075550892, 73.02696481420585, 0.25510086865823206]
            elif self.component == 'instrument':
                wt = [1.1214965070280278, 10.14041095890411, 9.409957627118644, 0.3444180215576292]
            elif self.component == 'instrument-verb':
                wt = [5.06264796142043, 22.5544921875, 21.095907928388748, 0.26948836905384216]
            elif self.component == 'instrument-target':
                wt = [11.01784813319879, 49.762343334599315, 46.64444642221431, 0.25855133464755714]
        elif self.dataset == 'prosta':
            if self.component == 'verb':
                wt = [1.578496832734121, 3.1925207756232687, 2.940051020408163, 0.36857885268838697]
            elif self.component == 'triplet':
                wt = [11.636131593874078, 16.577373737373737, 17.077627471383973, 0.2634920125230794]
        elif self.dataset == 'grasp':
            wt = [8.188049853372434, 14.442025862068965, 14.59298780487805, 0.26737236063009717]
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
            labels = torch.cat([sample.sc_v for sample in data_samples])
        elif self.component == 'triplet':
            labels = torch.cat([sample.sc_ivt for sample in data_samples])
        elif self.component == 'instrument-target':
            labels = torch.cat([sample.sc_it for sample in data_samples])
        elif self.component == 'instrument-verb':
            labels = torch.cat([sample.sc_iv for sample in data_samples])
        elif self.component == 'instrument':
            labels = torch.cat([sample.sc_i for sample in data_samples])
        return labels

    def loss(self, 
            inputs: torch.Tensor, 
            data_samples: SampleList,
            **kwargs) -> dict:
        state_logits = self(inputs)
        losses = dict()
        labels = self.get_labels(data_samples)
        class_state_weight = self.get_weights(inputs.device)

        log_probs = F.log_softmax(state_logits, dim=-1)   # [B, C, 4]

        # negative log likelihood for the target state of each class
        nll = -log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)  # [B, C]

        # pick the right weight for each (B, C) label
        class_state_weight = class_state_weight.unsqueeze(0).expand(labels.size(0), -1, -1)  # [B, C, 4]
        sample_weight = class_state_weight.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)  # [B, C]

        loss_sc = (nll * sample_weight).mean()

        losses[f'loss_sc_{self.component}'] = loss_sc
        losses['state_logits'] = state_logits

        return losses

    def predict(self, 
                inputs: torch.Tensor, 
                data_samples: SampleList,
                **kwargs) -> SampleList:
        
        state_logits = self(inputs)
        state_logits = state_logits.softmax(dim=-1)
        labels = self.get_labels(data_samples)

        for data_sample, logit, label in zip(data_samples, state_logits, labels):
            data_sample.set_pred_score(logit)
            data_sample.set_pred_label(label)

        return data_samples


    
from einops import rearrange

@MODELS.register_module()
class SCHead(BaseModule):
    """State Change Head to predict state change from the class embeddings."""
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 clip_len: int = 4,
                 num_states: int = 4,
                 num_clusters: int = 25,
                 num_classes: int = 10,
                 num_fc_layers: int = 1,
                 component: str = 'verb',
                 loss_state: dict = dict(type='CrossEntropyLoss', loss_weight=1.0),
                 init_cfg: Optional[ConfigType] = None,
                ) -> None:
        super().__init__(init_cfg=init_cfg)

        self.component = component
        self.clip_len = clip_len
        self.loss_state = loss_state
        self.get_loss_func()
        self.num_states = num_states
        self.num_classes = num_classes
        self.num_clusters = num_clusters
        out_channels = num_states * num_classes
        in_channels = in_channels * num_clusters
        if num_fc_layers == 1:
            self.fc_state = nn.Sequential(
                nn.LayerNorm(in_channels),
                nn.Linear(in_channels, out_channels))
        else:
            self.fc_state = nn.Sequential(
                nn.LayerNorm(in_channels),
                nn.Linear(in_channels, in_channels//2),
                nn.GELU(approximate='none'),
                nn.Linear(in_channels//2, out_channels))
            
    def get_loss_func(self):
        if self.loss_state['type'] == 'CrossEntropyLoss':
            self.loss_cls = MODELS.build(self.loss_state)
        elif self.loss_state['type'] == 'AsymmetricLoss':
            from timm.loss import AsymmetricLossSingleLabel
            self.loss_cls = AsymmetricLossSingleLabel(
                gamma_neg=self.loss_state['gamma_neg'],
                gamma_pos=self.loss_state['gamma_pos'],
                eps=1e-8,
            )
        else:
            raise ValueError(f'Loss function {self.loss_state["type"]} not supported')

    def forward(self, inputs: torch.Tensor, **kwargs) -> dict:
        inputs = rearrange(inputs, 'b n d -> b (n d)')
        out = self.fc_state(inputs)
        out = rearrange(out, 'b (c s) -> b c s', s=self.num_states)
        return out
    
    def get_weights(self, device: torch.device):
        if self.component == 'verb':
            wt = [1.9524793388429753, 9.218555417185554, 8.544632550981147, 0.30653020097173145]
        elif self.component == 'triplet':
            wt = [18.72154779969651, 77.67576075550892, 73.02696481420585, 0.25510086865823206]
        return torch.tensor(wt).to(device)
    
    def get_labels(self, data_samples: SampleList) -> torch.Tensor:
        if self.component == 'verb':
            labels = torch.cat([sample.sc_v for sample in data_samples])
        elif self.component == 'triplet':
            labels = torch.cat([sample.sc_ivt for sample in data_samples])
        elif self.component == 'tool_tissue':
            labels = torch.cat([sample.sc_it for sample in data_samples])
        elif self.component == 'tool':
            labels = torch.cat([sample.sc_i for sample in data_samples])
        return labels

    def loss(self, 
            inputs: torch.Tensor, 
            data_samples: SampleList,
            **kwargs) -> dict:
        
        state_logits = self(inputs)
        losses = dict()
        labels = self.get_labels(data_samples)
        class_weight = self.get_weights(inputs.device)

        if self.loss_state['type'] == 'CrossEntropyLoss':
            loss_sc = self.loss_cls(state_logits.permute(0, 2, 1), labels, weight=class_weight)
        elif self.loss_state['type'] == 'AsymmetricLoss':
            # all class together
            # loss_sc = self.loss_cls(state_logits.permute(0, 2, 1), labels) * self.loss_state['loss_weight']
            # per class separately computing loss and then averaging
            loss_sc = 0
            for c in range(self.num_classes):
                loss_sc_per_class = self.loss_cls(state_logits[:, c, :], labels[:, c])
                loss_sc = loss_sc + loss_sc_per_class
            loss_sc = loss_sc / self.num_classes
            loss_sc = loss_sc * self.loss_state['loss_weight']
        else:
            raise ValueError(f'Loss function {self.loss_state["type"]} not supported')

        losses[f'loss_sc_{self.component}'] = loss_sc
        losses['state_logits'] = state_logits

        return losses

    def predict(self, 
                inputs: torch.Tensor, 
                data_samples: SampleList,
                **kwargs) -> SampleList:
        
        state_logits = self(inputs)
        state_logits = state_logits.softmax(dim=-1)
        labels = self.get_labels(data_samples)

        for data_sample, logit, label in zip(data_samples, state_logits, labels):
            data_sample.set_pred_score(logit)
            data_sample.set_pred_label(label)

        return data_samples