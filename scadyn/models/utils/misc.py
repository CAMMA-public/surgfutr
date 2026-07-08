import torch
from torch import Tensor
import torch.nn as nn
from mmengine.runner import load_checkpoint
from mmengine.config import Config
from mmengine.registry import DefaultScope
from mmaction.registry import MODELS
import importlib

@MODELS.register_module()
class RNNFeaturePredictor(nn.Module):
    """Class for RNN Predictor with LayerNorm and Dummy (fixed 0s) input for unrolling."""
    def __init__(self, 
        in_channels: int, 
        hid_channels: int, 
        out_channels: int,
        num_layers: int = 1,
        model_type='LSTM'
        ):
        super().__init__()
        get_rnn = getattr(nn, model_type)
        self.rnn = get_rnn(in_channels, hid_channels, num_layers, batch_first=True)
        self.fc = nn.Linear(hid_channels, out_channels)

    def forward(self, 
                x: Tensor, 
                future_steps: int=4) -> Tensor:
        
        _, hidden_state = self.rnn(x)
        outs = torch.zeros(x.size(0), future_steps, self.fc.out_features).to(x.device)
        feat_input = x[:, -1, :].unsqueeze(1)
        for i in range(future_steps):
            out, hidden_state = self.rnn(feat_input, hidden_state)
            out = self.fc(out[:, -1, :]) # b x d
            outs[:, i, :] = out.squeeze(1) # b x d
            feat_input = out.unsqueeze(1)
        return outs
    
def get_model_from_cfg(cfg_path,
                       pretrained: bool = False,
                       model_path: str = '',
                       module_name: str = 'all',
                       **kwargs):
    package = cfg_path.split('::')[0]
    with DefaultScope.overwrite_default_scope(package):  # type: ignore
        cfg_path = cfg_path.split('::')[1]
        cfg_path = 'configs/' + cfg_path
        cfg = Config.fromfile(cfg_path)
        if 'data_preprocessor' in cfg:
            cfg.model.data_preprocessor = cfg.data_preprocessor
        models_module = importlib.import_module(f'{package}.utils')
        models_module.register_all_modules()  # type: ignore
        model = MODELS.build(cfg.model, default_args=kwargs)
        if pretrained and model_path is not None:
            load_checkpoint(model, model_path)
            # Hack to use pretrained weights.
            # If we do not set _is_init here, Runner will call
            # `model.init_weights()` to overwrite the pretrained model.
            model._is_init = True

        if module_name == 'all':
            return model
        elif module_name == 'backbone':
            return model.backbone
        else:
            raise ValueError(f"module_name {module_name} not found in model")

import numpy as np
import matplotlib.pyplot as plt

def visualize_clustering(input_clip, 
                        assignment_matrix, 
                        batch_idx=0, 
                        tubelet_size=2, 
                        patch_size=16,
                        ignore_even_frames=False, 
                        return_fig=False,
                        mean=[114.75, 114.75, 114.75], 
                        std=[57.375, 57.375, 57.375],
                        overlay_alpha: float = 0.8,
                        overlay_brightness: float = 0.75,
                        overlay_contrast: float = 1.15):
    """
    Visualize clustering of video features, with denormalization of input clip.
    
    Args:
    - input_clip (torch.Tensor): Input video clip of shape (B, C, T, H, W)
    - assignment_matrix (torch.Tensor): Assignment matrix of shape (B, 784, K)
    - tubelet_size (int): Temporal size of each tubelet
    - patch_size (int): Spatial size of each patch
    - mean (list): Mean values used for normalization
    - std (list): Standard deviation values used for normalization
    
    Returns:
    - If return_fig is True: list of numpy arrays (one per batch) of shape (H, W*T, C)
    - Otherwise: None (saves the visualization per batch)
    """
    B, C, T, H, W = input_clip.shape
    _, N, K = assignment_matrix.shape
    
    # Build a distinct color palette so clusters stand out
    def _generate_distinct_rgba_colors(num_colors: int):
        from matplotlib import colors as mpl_colors
        # Kelly's 20 distinct colors (excluding white/black), hex -> RGB
        kelly_hex = [
            '#FFB300', '#803E75', '#FF6800', '#A6BDD7', '#C10020',
            '#CEA262', '#817066', '#007D34', '#F6768E', '#00538A',
            '#FF7A5C', '#53377A', '#FF8E00', '#B32851', '#F4C800',
            '#7F180D', '#93AA00', '#593315', '#F13A13', '#232C16'
        ]
        kelly_rgb = np.array([mpl_colors.to_rgb(h) for h in kelly_hex], dtype=float)
        kelly_rgba = np.concatenate([kelly_rgb, np.ones((kelly_rgb.shape[0], 1))], axis=1)

        if num_colors <= kelly_rgba.shape[0]:
            return kelly_rgba[:num_colors]

        # If more colors are needed, add hues with farthest-point sampling on hue circle
        # Initialize with Kelly set
        base_rgb = kelly_rgb.copy()
        base_hsv = mpl_colors.rgb_to_hsv(base_rgb.reshape(-1, 1, 3)).reshape(-1, 3)
        chosen_hues = list(base_hsv[:, 0])

        def circular_hue_distance(a, b):
            # distances in [0, 0.5]
            d = np.abs(a - b)
            return np.minimum(d, 1.0 - d)

        target_total = num_colors
        min_hue_delta = 0.10  # enforce at least ~36 degrees apart when possible
        sat = 0.75
        val = 0.85

        while len(chosen_hues) < target_total:
            candidates = np.linspace(0.0, 1.0, 360, endpoint=False)
            # compute min distance to existing hues for each candidate
            if len(chosen_hues) > 0:
                dists = np.min([circular_hue_distance(candidates, h) for h in chosen_hues], axis=0)
            else:
                dists = np.ones_like(candidates)
            # Prefer candidates beyond min_hue_delta; fallback to max distance
            mask = dists >= min_hue_delta
            if np.any(mask):
                idx = np.argmax(dists * mask)
            else:
                idx = np.argmax(dists)
            chosen_hues.append(candidates[idx])

        # Build final RGB from chosen hues (keep original Kelly first to preserve strong separation)
        extra_hues = chosen_hues[len(kelly_rgb):]
        extra_hsv = np.stack([np.array(extra_hues), np.full(len(extra_hues), sat), np.full(len(extra_hues), val)], axis=1)
        extra_rgb = mpl_colors.hsv_to_rgb(extra_hsv)
        all_rgb = np.concatenate([kelly_rgb, extra_rgb], axis=0)[:num_colors]
        all_rgba = np.concatenate([all_rgb, np.ones((all_rgb.shape[0], 1))], axis=1)
        return all_rgba

    colors = _generate_distinct_rgba_colors(K)
    
    # Get the cluster assignments
    cluster_ids = torch.argmax(assignment_matrix, dim=2)  # Shape: (B, 784)
    
    # Reshape cluster_ids to match the spatial-temporal structure
    cluster_ids = cluster_ids.reshape(B, T // tubelet_size, H // patch_size, W // patch_size)
    
    # Denormalize the input clip
    mean = torch.tensor(mean).view(1, 3, 1, 1, 1).to(input_clip.device)
    std = torch.tensor(std).view(1, 3, 1, 1, 1).to(input_clip.device)
    denormalized_clip = input_clip * std + mean
    denormalized_clip = torch.clamp(denormalized_clip, 0, 255).byte()

    # ignore even frames
    # if ignore_even_frames:
    #     denormalized_clip = denormalized_clip[:, :, ::2, :, :]
    #     # cluster ids are already T/2
    #     T = T // 2
    
    # If returning figures, collect strips per batch; otherwise save per-batch figures
    returned_strips = []
    import os
    from datetime import datetime
    os.makedirs('visualization', exist_ok=True)
    base_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Determine which batch indices to process
    batch_indices = list(range(B))
    
    for b in batch_indices:
        # Create a figure with subplots for each frame
        fig, axes = plt.subplots(2, T // 2, figsize=(20, 8))
        axes = axes.flatten()
        blended_frames = []
        for t in range(T):
            ax = axes[t]
            
            # Get the frame and its cluster assignments
            frame = denormalized_clip[b, :, t, :, :].permute(1, 2, 0).cpu().numpy()
            frame_clusters = cluster_ids[b, t // tubelet_size, :, :]
            
            # Create a color-coded overlay
            overlay = colors[frame_clusters.cpu().numpy()]
            
            # Resize overlay to match frame size
            overlay_resized = np.repeat(np.repeat(overlay, patch_size, axis=0), patch_size, axis=1)
            
            # create a blended image
            # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = frame.copy()
            # Adjust overlay brightness and contrast before blending for visibility
            # contrast: (x-0.5)*c + 0.5; brightness: multiply
            overlay_rgb = overlay_resized[..., :3]
            overlay_rgb = (overlay_rgb - 0.5) * overlay_contrast + 0.5
            overlay_rgb = np.clip(overlay_rgb * overlay_brightness, 0.0, 1.0)
            overlay_resized[..., 3] = overlay_alpha
            overlay_resized_rgb = (overlay_rgb * 255).astype(np.uint8)
            alpha = overlay_resized[..., 3:4]
            blended_frame = frame_rgb * (1 - alpha) + overlay_resized_rgb * alpha
            blended_frames.append(blended_frame.astype(np.uint8))
            
            # Display the frame
            ax.imshow(frame)
            
            # Add the color-coded overlay with transparency
            ax.imshow(overlay_resized, alpha=0.5)
            
            ax.set_title(f'Frame {t+1}')
            ax.axis('off')
        
        plt.tight_layout()

        if return_fig:
            # convert to numpy array t x h x w x c
            clip = np.stack(blended_frames, axis=0)
            
            # Rearrange frames into a horizontal strip
            T_clip, H_clip, W_clip, C_clip = clip.shape
            strip_height = H_clip
            strip_width = W_clip * T_clip
            strip = np.zeros((strip_height, strip_width, C_clip), dtype=np.uint8)
            
            for t in range(T_clip):
                strip[:, t*W_clip:(t+1)*W_clip, :] = clip[t]
            
            returned_strips.append(strip)
        else:
            # save to visualization folder with timestamp and batch index
            # folder = 'cholect50'
            folder = 'grasp'
            plt.savefig(f'visualization/{folder}/clustering_{base_timestamp}_b{b}.png')
        plt.close(fig)

    if return_fig:
        return returned_strips

        