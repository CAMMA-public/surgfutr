from typing import Optional

import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmaction.registry import MODELS
from mmaction.utils import ConfigType, SampleList


@MODELS.register_module()
class TemporalPhaseHead(BaseModule):
    """Simple phase-recognition head for [B, T, D] or [B, D] features."""

    def __init__(
        self,
        input_dim: int = 2048,
        in_channels: Optional[int] = None,
        num_classes: int = 7,
        label_type: str = "auto",
        feat_agg: str = "mean",
        loss_cls: dict = dict(type="CrossEntropyLoss", loss_weight=1.0),
        class_weights: Optional[list] = None,
        init_cfg: Optional[ConfigType] = None,
        **kwargs,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        # Accept both `input_dim` and inherited mmaction-style `in_channels`.
        self.input_dim = int(in_channels if in_channels is not None else input_dim)
        self.num_classes = int(num_classes)
        self.label_type = str(label_type)
        self.feat_agg = feat_agg
        self.loss_cls = MODELS.build(loss_cls)
        self.fc_cls = nn.Linear(self.input_dim, self.num_classes)
        self.class_weights = class_weights

        # Defaults from current datasets.
        self.phase_class_weights = [
            3.420485175202156, 0.3570624648283624, 1.538181818181818,
            0.4830605253140464, 3.2790697674418605, 1.58625, 3.9780564263322886
        ]
        self.step_class_weights = [
            0.17425731760594146, 0.6384553821528611, 1.0336896663427275,
            1.1339729921819475, 1.7668881506090808, 1.7465790914066777,
            3.896214896214896, 3.4534632034632033, 0.9044784580498866,
            2.4508448540706604, 2.6198686371100166, 13.813852813852813,
            12.662698412698413, 1.5665193912616593, 1.1091414668057005,
            1.0479474548440066, 2.57546408393866, 1.101104209799862,
            0.38566594150350497, 0.8258281573498965, 1.9234478601567209
        ]

    def _aggregate(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim == 3:
            if self.feat_agg == "last":
                return inputs[:, -1]
            return inputs.mean(dim=1)
        if inputs.ndim == 2:
            return inputs
        raise ValueError(f"TemporalPhaseHead expects [B,T,D] or [B,D], got {tuple(inputs.shape)}")

    def _get_labels(self, data_samples: SampleList, device: torch.device) -> torch.Tensor:
        labels = []
        for sample in data_samples:
            lbl = None
            if hasattr(sample, "gt_label"):
                lbl = sample.gt_label
            elif hasattr(sample, "label"):
                lbl = sample.label
            elif hasattr(sample, "cl_v"):
                lbl = sample.cl_v
            elif self.label_type in ("phase", "auto") and hasattr(sample, "phase"):
                lbl = sample.phase
            elif self.label_type in ("step", "auto") and hasattr(sample, "step"):
                lbl = sample.step
            elif hasattr(sample, "metainfo") and isinstance(sample.metainfo, dict):
                if "gt_label" in sample.metainfo:
                    lbl = sample.metainfo["gt_label"]
                elif "label" in sample.metainfo:
                    lbl = sample.metainfo["label"]
                elif "cl_v" in sample.metainfo:
                    lbl = sample.metainfo["cl_v"]
                elif self.label_type in ("phase", "auto") and "phase" in sample.metainfo:
                    lbl = sample.metainfo["phase"]
                elif self.label_type in ("step", "auto") and "step" in sample.metainfo:
                    lbl = sample.metainfo["step"]
            if lbl is None:
                raise ValueError("TemporalPhaseHead requires labels (`gt_label`/`label`/`cl_v` or dataset key like `phase`/`step`).")
            t = torch.as_tensor(lbl, device=device).reshape(-1)
            # Support scalar class-id labels and one-hot / score vectors.
            labels.append((t.argmax() if t.numel() > 1 else t[0]).long())
        return torch.stack(labels, dim=0)

    def _get_class_weights(self, device: torch.device) -> Optional[torch.Tensor]:
        # Config override has highest priority.
        if self.class_weights is not None:
            w = torch.tensor(self.class_weights, device=device, dtype=torch.float32)
            return w if w.numel() == self.num_classes else None

        # Auto-select by task hint and class count.
        if self.label_type == "phase" or (self.label_type == "auto" and self.num_classes == len(self.phase_class_weights)):
            return torch.tensor(self.phase_class_weights, device=device, dtype=torch.float32)
        if self.label_type == "step" or (self.label_type == "auto" and self.num_classes == len(self.step_class_weights)):
            return torch.tensor(self.step_class_weights, device=device, dtype=torch.float32)
        return None

    def forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self._aggregate(inputs)
        return self.fc_cls(x)

    def loss(self, inputs: torch.Tensor, data_samples: SampleList, **kwargs) -> dict:
        logits = self(inputs)
        labels = self._get_labels(data_samples, device=inputs.device)
        cls_weights = self._get_class_weights(device=inputs.device)
        if cls_weights is not None:
            loss = self.loss_cls(logits, labels, weight=cls_weights)
        else:
            loss = self.loss_cls(logits, labels)
        return {"loss_cls_phase": loss}

    def predict(self, inputs: torch.Tensor, data_samples: SampleList, **kwargs) -> SampleList:
        logits = self(inputs)
        scores = logits.softmax(dim=-1)
        labels = self._get_labels(data_samples, device=inputs.device)
        pred_labels = scores.argmax(dim=-1)
        for data_sample, score, pred_label, gt_label in zip(data_samples, scores, pred_labels, labels):
            data_sample.set_pred_score(score)
            data_sample.set_pred_label(pred_label.unsqueeze(0))
            # AccMetric expects `gt_label`; fill it only when absent.
            if not hasattr(data_sample, "gt_label"):
                if hasattr(data_sample, "set_gt_label"):
                    data_sample.set_gt_label(gt_label.unsqueeze(0))
                else:
                    data_sample.set_field(gt_label.unsqueeze(0), "gt_label")
        return data_samples
