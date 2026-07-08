import cv2
import io
import os.path as osp
import random
from typing import Dict, List, Optional, Union, Sequence, Tuple
from numbers import Number
import decord
import mmcv
import numpy as np
import torch
from mmcv.transforms import BaseTransform
from mmengine.fileio import FileClient
from mmcv.transforms import BaseTransform, to_tensor
from mmengine.utils import digit_version
import mmengine
from mmcv.transforms.utils import cache_randomness
from mmaction.registry import TRANSFORMS
from mmaction.structures import ActionDataSample
from mmaction.structures.action_data_sample import format_label
from mmaction.datasets.transforms.processing import RandomCrop
from mmaction.datasets.transforms.formatting import PackActionInputs, FormatShape

LABEL_TYPE = Union[torch.Tensor, np.ndarray, Sequence, int]


class CustomActionDataSample(ActionDataSample):
    def set_value_x(self, value: LABEL_TYPE, x) -> 'CustomActionDataSample':
        """Set ``pred_label``."""
        self.set_field(format_label(value), f'{x}')

@TRANSFORMS.register_module()
class PackTeacherStudentInputsFrame(PackActionInputs):
    def __init__(
            self,
            enable_ts: bool = False,
            enable_futr: bool = True,
            extra_keys: Sequence[str] = (),
            **kwargs) -> None:
        super().__init__(**kwargs)
        self.enable_ts = enable_ts
        self.enable_futr = enable_futr
        self.extra_keys = extra_keys

    def transform(self, results: Dict) -> Dict:
        packed_results = dict()
        if 'imgs' in results:
            if not self.enable_ts:
                imgs = results['imgs']
                packed_results['inputs'] = to_tensor(imgs)
        elif 'features' in results:
            features = results['features']
            packed_results['inputs'] = to_tensor(features)

        if 'future_imgs' in results and not self.enable_ts:
            packed_results['inputs'] = torch.cat([packed_results['inputs'], 
                                                  to_tensor(results['future_imgs'])], dim=0) # noqa
            
        if self.enable_ts:
            if not self.enable_futr:
                packed_results['inputs'] = torch.cat([
                    to_tensor(results['imgs_s']),
                    to_tensor(results['imgs_t'])], dim=0)
            else:
                packed_results['inputs'] = torch.cat([
                    to_tensor(results['imgs_s']),
                    to_tensor(results['future_imgs_s']),
                    to_tensor(results['imgs_t']),
                    to_tensor(results['future_imgs_t'])], dim=0)

        data_sample = CustomActionDataSample()

        if 'label' in results:
            data_sample.set_gt_label(results['label'])
        if 'mask' in results: # NOTE: add mask for the variable input shape ..
            data_sample.set_data({"mask":results["mask"]})
        for k in self.extra_keys:
            if k in results:
                data_sample.set_data({k: results[k]})

        # Set meta keys
        img_meta = {k: results[k] for k in self.meta_keys if k in results}
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample
        return packed_results
    
@TRANSFORMS.register_module()
class FormatTeacherStudentShapeFrame(FormatShape):
    """Format final imgs shape to the given input_format.

    Required keys:

        - imgs (optional)
        - heatmap_imgs (optional)
        - modality (optional)
        - num_clips
        - clip_len

    Modified Keys:

        - imgs

    Added Keys:

        - input_shape
        - heatmap_input_shape (optional)

    Args:
        input_format (str): Define the final data format.
        collapse (bool): To collapse input_format N... to ... (NCTHW to CTHW,
            etc.) if N is 1. Should be set as True when training and testing
            detectors. Defaults to False.
    """

    def __init__(self, 
                 enable_ts: bool = False,
                 enable_futr: bool = True,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.enable_ts = enable_ts
        self.enable_futr = enable_futr

    def process_imgs(self, results, key):
        imgs = results[key]
        num_clips = results['num_clips']
        clip_len = results['clip_len']
        if isinstance(clip_len, dict):
            clip_len = clip_len['RGB']

        #print("results['clip_len']", results['clip_len'])
        imgs = imgs.reshape((-1, num_clips, clip_len) + imgs.shape[1:])
        # N_crops x N_clips x T x H x W x C
        imgs = np.transpose(imgs, (0, 1, 5, 2, 3, 4))
        # N_crops x N_clips x C x T x H x W
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        # M' x C x T x H x W
        # M' = N_crops x N_clips
        results[key] = imgs
        results['input_shape'] = imgs.shape
        return results

    def transform(self, results: Dict) -> Dict:
        """Performs the FormatShape formatting.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if not isinstance(results['imgs'], np.ndarray):
            results['imgs'] = np.array(results['imgs'])

        if 'future_imgs' in results:
            if not isinstance(results['future_imgs'], np.ndarray):
                results['future_imgs'] = np.array(results['future_imgs'])

        if self.enable_ts:
            if not isinstance(results['imgs_s'], np.ndarray):
                results['imgs_s'] = np.array(results['imgs_s'])

            if not isinstance(results['imgs_t'], np.ndarray):
                results['imgs_t'] = np.array(results['imgs_t'])

            if self.enable_futr:
                if not isinstance(results['future_imgs_s'], np.ndarray):
                    results['future_imgs_s'] = np.array(results['future_imgs_s'])

                if not isinstance(results['future_imgs_t'], np.ndarray):
                    results['future_imgs_t'] = np.array(results['future_imgs_t'])

        if self.input_format == 'NCTHW':
            if 'imgs' in results and not self.enable_ts:
                results = self.process_imgs(results, 'imgs')
            
            if 'imgs_s' in results and self.enable_ts:
                results = self.process_imgs(results, 'imgs_s')

            if 'imgs_t' in results and self.enable_ts:
                results = self.process_imgs(results, 'imgs_t')

            if self.enable_futr:
                if 'future_imgs' in results and not self.enable_ts:
                    results = self.process_imgs(results, 'future_imgs')

                if 'future_imgs_s' in results and self.enable_ts:
                    results = self.process_imgs(results, 'future_imgs_s')

                if 'future_imgs_t' in results and self.enable_ts:
                    results = self.process_imgs(results, 'future_imgs_t')

        if self.collapse:
            assert results['imgs'].shape[0] == 1
            results['imgs'] = results['imgs'].squeeze(0)
            results['input_shape'] = results['imgs'].shape

        return results
    

@TRANSFORMS.register_module()
class RawFrameDecodeFrame(BaseTransform):
    import copy as cp
    """Load and decode frames with given indices.

    Required Keys:

    - frame_dir
    - filename_tmpl
    - frame_inds
    - modality
    - offset (optional)

    Added Keys:

    - img
    - img_shape
    - original_shape

    Args:
        io_backend (str): IO backend where frames are stored.
            Defaults to ``'disk'``.
        decoding_backend (str): Backend used for image decoding.
            Defaults to ``'cv2'``.
    """

    def __init__(self,
                 io_backend: str = 'disk',
                 decoding_backend: str = 'cv2',
                 **kwargs) -> None:
        self.io_backend = io_backend
        self.decoding_backend = decoding_backend
        self.kwargs = kwargs
        self.file_client = None

    def transform(self, results: dict) -> dict:
        """Perform the ``RawFrameDecode`` to pick frames given indices.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        mmcv.use_backend(self.decoding_backend)

        directory = results['frame_dir']
        filename_tmpl = results['filename_tmpl']
        modality = results['modality']

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)

        imgs = list()

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        offset = results.get('offset', 0)

        cache = {}
        for i, frame_idx in enumerate(results['frame_inds']):
            # Avoid loading duplicated frames
            if frame_idx in cache:
                imgs.append(cp.deepcopy(imgs[cache[frame_idx]]))
                continue
            else:
                cache[frame_idx] = i

            frame_idx += offset
            if modality == 'RGB':
                filepath = osp.join(directory, filename_tmpl.format(frame_idx))
                img_bytes = self.file_client.get(filepath)
                # Get frame with channel order RGB directly.
                cur_frame = mmcv.imfrombytes(img_bytes, channel_order='rgb')
                imgs.append(cur_frame)
            elif modality == 'Flow':
                x_filepath = osp.join(directory,
                                      filename_tmpl.format('x', frame_idx))
                y_filepath = osp.join(directory,
                                      filename_tmpl.format('y', frame_idx))
                x_img_bytes = self.file_client.get(x_filepath)
                x_frame = mmcv.imfrombytes(x_img_bytes, flag='grayscale')
                y_img_bytes = self.file_client.get(y_filepath)
                y_frame = mmcv.imfrombytes(y_img_bytes, flag='grayscale')
                imgs.append(np.stack([x_frame, y_frame], axis=-1))
            else:
                raise NotImplementedError

        results['imgs'] = imgs
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]

        # we resize the gt_bboxes and proposals to their real scale
        if 'gt_bboxes' in results:
            h, w = results['img_shape']
            scale_factor = np.array([w, h, w, h])
            gt_bboxes = results['gt_bboxes']
            gt_bboxes = (gt_bboxes * scale_factor).astype(np.float32)
            results['gt_bboxes'] = gt_bboxes
            if 'proposals' in results and results['proposals'] is not None:
                proposals = results['proposals']
                proposals = (proposals * scale_factor).astype(np.float32)
                results['proposals'] = proposals

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'io_backend={self.io_backend}, '
                    f'decoding_backend={self.decoding_backend})')
        return repr_str

import copy as cp
@TRANSFORMS.register_module()
class LoadCurFutrFrame(BaseTransform):
    """Load frames from future clip & current clip + gtboxes."""
    def __init__(self,
                 io_backend: str = 'disk',
                 num_threads: int = 1,
                 clip_len: int = 8,
                 num_clips: int = 1,
                 enable_futr=True,
                 decoding_backend: str = 'cv2',
                 dataset_name: str = 'grasp',
                 **kwargs) -> None:
        self.io_backend = io_backend
        self.num_threads = num_threads
        self.clip_len = clip_len
        self.num_clips = num_clips
        self.kwargs = kwargs
        self.file_client = None
        self.enable_futr = enable_futr
        self.dataset_name = dataset_name
        self.decoding_backend = decoding_backend

    def transform(self, results: Dict) -> Dict:
        # container = self._get_video_reader(results['filename'])
        mmcv.use_backend(self.decoding_backend)

        dataset_name = self.dataset_name

        # collect the frame_inds NOTE: fixed 
        frame_inds = results['frame_inds'].numpy().tolist()
        results['frame_inds'] = frame_inds
        results['clip_len'] = self.clip_len
        results['num_clips'] = self.num_clips

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)
        imgs = list()
        
        cache = {}
        for i, frame_idx in enumerate(results['frame_inds']):
            if frame_idx in cache:
                imgs.append(cp.deepcopy(imgs[cache[frame_idx]]))
                continue
            else:
                cache[frame_idx] = i
            
            # if 'grasp' in results['data_prefix']:
            #     subfolder = "frames"
            #     frame_idx_str = f"{frame_idx:05d}.jpg"
            # else:
            #     subfolder = "videos"
            #     frame_idx_str = f"{frame_idx}.jpg"

            if dataset_name == 'grasp':
                subfolder = "frames"
                frame_idx_str = f"{frame_idx:05d}.jpg"
            elif dataset_name == 'prosta':
                subfolder = "videos"
                frame_idx_str = f"{frame_idx}.jpg"
            elif dataset_name == 'mbp':
                subfolder = "frames"
                frame_idx_str = f"{results['video_id']}_{frame_idx:08d}.jpg"
            else:
                raise ValueError(f"Dataset name {dataset_name} not supported")

            frame_idx_str = osp.join(results['video_id'], frame_idx_str)

            filepath = osp.join(results['data_prefix'], subfolder, frame_idx_str)
            img_bytes = self.file_client.get(filepath)
            cur_frame = mmcv.imfrombytes(img_bytes, channel_order='rgb')
            imgs.append(cur_frame)

        results['imgs'] = imgs
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]

        # print("total frames loaded: ", len(imgs), len(results['frame_inds']), results['frame_inds'])

        if self.enable_futr:
            future_frame_inds = results['future_frame_inds'].numpy().tolist()
            results['future_frame_inds'] = future_frame_inds

            cache = {}
            futr_imgs = list()
            for i, frame_idx in enumerate(results['future_frame_inds']):
                if frame_idx in cache:
                    futr_imgs.append(cp.deepcopy(futr_imgs[cache[frame_idx]]))
                    continue
                else:
                    cache[frame_idx] = i

                # if 'grasp' in results['data_prefix']:
                #     subfolder = "frames"
                #     frame_idx_str = f"{frame_idx:05d}.jpg"
                # else:
                #     subfolder = "videos"
                #     frame_idx_str = f"{frame_idx}.jpg"
                # frame_idx_str = osp.join(results['video_id'], frame_idx_str)

                if dataset_name == 'grasp':
                    subfolder = "frames"
                    frame_idx_str = f"{frame_idx:05d}.jpg"
                elif dataset_name == 'prosta':
                    subfolder = "videos"
                    frame_idx_str = f"{frame_idx}.jpg"
                elif dataset_name == 'mbp':
                    subfolder = "frames"
                    frame_idx_str = f"{results['video_id']}_{frame_idx:08d}.jpg"
                else:
                    raise ValueError(f"Dataset name {dataset_name} not supported")
                
                frame_idx_str = osp.join(results['video_id'], frame_idx_str)
                filepath = osp.join(results['data_prefix'], subfolder, frame_idx_str)
                img_bytes = self.file_client.get(filepath)
                cur_frame = mmcv.imfrombytes(img_bytes, channel_order='rgb')
                futr_imgs.append(cur_frame)

            # print("total future frames loaded: ", len(futr_imgs), len(results['future_frame_inds']), results['future_frame_inds'])

            future_clip_outputs = dict()
            future_clip_outputs['future_imgs'] = futr_imgs
            results.update(future_clip_outputs)

        return results

    def __repr__(self) -> str:
        repr_str = (f'{self.__class__.__name__}('
                    f'io_backend={self.io_backend}, '
                    f'num_threads={self.num_threads})')
        return repr_str

@TRANSFORMS.register_module()
class ResizeCurFutrFrame(BaseTransform):
    def __init__(self,
                 scale,
                 keep_ratio=True,
                 interpolation='bilinear',
                 input_key='s',
                 out_key='s',
                 enable_futr=True):
        if isinstance(scale, float):
            if scale <= 0:
                raise ValueError(f'Invalid scale {scale}, must be positive.')
        elif isinstance(scale, tuple):
            max_long_edge = max(scale)
            max_short_edge = min(scale)
            if max_short_edge == -1:
                # assign np.inf to long edge for rescaling short edge later.
                scale = (np.inf, max_long_edge)
        else:
            raise TypeError(
                f'Scale must be float or tuple of int, but got {type(scale)}')
        self.scale = scale
        self.keep_ratio = keep_ratio
        self.interpolation = interpolation
        self.out_key = out_key
        self.input_key = input_key
        self.enable_futr = enable_futr

    def _resize_imgs(self, imgs, new_w, new_h):
        """Static method for resizing keypoint."""
        return [
            mmcv.imresize(
                img, (new_w, new_h), interpolation=self.interpolation)
            for img in imgs
        ]

    def transform(self, results):
        if 'scale_factor' not in results:
            results['scale_factor'] = np.array([1, 1], dtype=np.float32)
        img_h, img_w = results['img_shape']

        if self.keep_ratio:
            new_w, new_h = mmcv.rescale_size((img_w, img_h), self.scale)
        else:
            new_w, new_h = self.scale

        self.scale_factor = np.array([new_w / img_w, new_h / img_h], dtype=np.float32)

        results['img_shape'] = (new_h, new_w)
        results['keep_ratio'] = self.keep_ratio
        results['scale_factor'] = results['scale_factor'] * self.scale_factor

        img_input_key = f'imgs_{self.input_key}' if self.input_key != '' else 'imgs'
        img_out_key = f'imgs_{self.out_key}' if self.out_key != '' else 'imgs'
        results[img_out_key] = self._resize_imgs(results[img_input_key], new_w, new_h)

        if self.enable_futr:
            futr_img_input_key = f'future_imgs_{self.input_key}' if self.input_key != '' else 'future_imgs'
            futr_img_out_key = f'future_imgs_{self.out_key}' if self.out_key != '' else 'future_imgs'
            results[futr_img_out_key] = self._resize_imgs(results[futr_img_input_key], new_w, new_h)
        
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'scale={self.scale}, keep_ratio={self.keep_ratio}, '
                    f'interpolation={self.interpolation}')
        return repr_str

@TRANSFORMS.register_module()
class FlipCurFutrFrame(BaseTransform):
    _directions = ['horizontal', 'vertical']

    def __init__(self,
                 flip_ratio=0.5,
                 direction='horizontal',
                 flip_label_map=None,
                 left_kp=None,
                 right_kp=None,
                 input_key='s',
                 out_key='s',
                 lazy=False,
                 enable_futr=True):
        if direction not in self._directions:
            raise ValueError(f'Direction {direction} is not supported. '
                             f'Currently support ones are {self._directions}')
        self.flip_ratio = flip_ratio
        self.direction = direction
        self.flip_label_map = flip_label_map
        self.left_kp = left_kp
        self.right_kp = right_kp
        self.lazy = lazy
        self.input_key = input_key
        self.out_key = out_key
        self.enable_futr = enable_futr

    def _flip_imgs(self, imgs, modality):
        """Utility function for flipping images."""
        imgs = [mmcv.imflip(img, direction=self.direction) for img in imgs]
        lt = len(imgs)
        if modality == 'Flow':
            # The 1st frame of each 2 frames is flow-x
            for i in range(0, lt, 2):
                imgs[i] = mmcv.iminvert(imgs[i])
        return imgs

    def transform(self, results):
        modality = results['modality']
        flip = np.random.rand() < self.flip_ratio

        results['flip'] = flip
        results['flip_direction'] = self.direction

        if flip:
            img_input_key = f'imgs_{self.input_key}' if self.input_key != '' else 'imgs'
            img_out_key = f'imgs_{self.out_key}' if self.out_key != '' else 'imgs'
            results[img_out_key] = self._flip_imgs(results[img_input_key], modality)

            if self.enable_futr:
                futr_img_input_key = f'future_imgs_{self.input_key}' if self.input_key != '' else 'future_imgs'
                futr_img_out_key = f'future_imgs_{self.out_key}' if self.out_key != '' else 'future_imgs'
                results[futr_img_out_key] = self._flip_imgs(results[futr_img_input_key], modality)

        return results
    

@TRANSFORMS.register_module()
class RandomResizedCropCurFutrFrame(RandomCrop):
    """Random crop that specifics the area and height-weight ratio range.

    Required keys in results are "img_shape", "crop_bbox", "imgs" (optional),
    "keypoint" (optional), added or modified keys are "imgs", "keypoint",
    "crop_bbox" and "lazy"; Required keys in "lazy" are "flip", "crop_bbox",
    added or modified key is "crop_bbox".

    Args:
        area_range (Tuple[float]): The candidate area scales range of
            output cropped images. Default: (0.08, 1.0).
        aspect_ratio_range (Tuple[float]): The candidate aspect ratio range of
            output cropped images. Default: (3 / 4, 4 / 3).
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    def __init__(self,
                 area_range=(0.08, 1.0),
                 aspect_ratio_range=(3 / 4, 4 / 3),
                 lazy=False,
                 input_key='s',
                 out_key='s',
                 enable_futr=True):
        self.area_range = area_range
        self.aspect_ratio_range = aspect_ratio_range
        self.lazy = lazy
        self.input_key = input_key
        self.out_key = out_key
        self.enable_futr = enable_futr
        if not mmengine.is_tuple_of(self.area_range, float):
            raise TypeError(f'Area_range must be a tuple of float, '
                            f'but got {type(area_range)}')
        if not mmengine.is_tuple_of(self.aspect_ratio_range, float):
            raise TypeError(f'Aspect_ratio_range must be a tuple of float, '
                            f'but got {type(aspect_ratio_range)}')

    @staticmethod
    def get_crop_bbox(img_shape,
                      area_range,
                      aspect_ratio_range,
                      max_attempts=10):
        """Get a crop bbox given the area range and aspect ratio range.

        Args:
            img_shape (Tuple[int]): Image shape
            area_range (Tuple[float]): The candidate area scales range of
                output cropped images. Default: (0.08, 1.0).
            aspect_ratio_range (Tuple[float]): The candidate aspect
                ratio range of output cropped images. Default: (3 / 4, 4 / 3).
                max_attempts (int): The maximum of attempts. Default: 10.
            max_attempts (int): Max attempts times to generate random candidate
                bounding box. If it doesn't qualified one, the center bounding
                box will be used.
        Returns:
            (list[int]) A random crop bbox within the area range and aspect
            ratio range.
        """
        assert 0 < area_range[0] <= area_range[1] <= 1
        assert 0 < aspect_ratio_range[0] <= aspect_ratio_range[1]

        img_h, img_w = img_shape
        area = img_h * img_w

        min_ar, max_ar = aspect_ratio_range
        aspect_ratios = np.exp(
            np.random.uniform(
                np.log(min_ar), np.log(max_ar), size=max_attempts))
        target_areas = np.random.uniform(*area_range, size=max_attempts) * area
        candidate_crop_w = np.round(np.sqrt(target_areas *
                                            aspect_ratios)).astype(np.int32)
        candidate_crop_h = np.round(np.sqrt(target_areas /
                                            aspect_ratios)).astype(np.int32)

        for i in range(max_attempts):
            crop_w = candidate_crop_w[i]
            crop_h = candidate_crop_h[i]
            if crop_h <= img_h and crop_w <= img_w:
                x_offset = random.randint(0, img_w - crop_w)
                y_offset = random.randint(0, img_h - crop_h)
                return x_offset, y_offset, x_offset + crop_w, y_offset + crop_h

        # Fallback
        crop_size = min(img_h, img_w)
        x_offset = (img_w - crop_size) // 2
        y_offset = (img_h - crop_size) // 2
        return x_offset, y_offset, x_offset + crop_size, y_offset + crop_size

    def transform(self, results):
        """Performs the RandomResizeCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        img_h, img_w = results['img_shape']

        left, top, right, bottom = self.get_crop_bbox(
            (img_h, img_w), self.area_range, self.aspect_ratio_range)
        new_h, new_w = bottom - top, right - left

        if 'crop_quadruple' not in results:
            results['crop_quadruple'] = np.array(
                [0, 0, 1, 1],  # x, y, w, h
                dtype=np.float32)

        x_ratio, y_ratio = left / img_w, top / img_h
        w_ratio, h_ratio = new_w / img_w, new_h / img_h

        old_crop_quadruple = results['crop_quadruple']
        old_x_ratio, old_y_ratio = old_crop_quadruple[0], old_crop_quadruple[1]
        old_w_ratio, old_h_ratio = old_crop_quadruple[2], old_crop_quadruple[3]
        new_crop_quadruple = [
            old_x_ratio + x_ratio * old_w_ratio,
            old_y_ratio + y_ratio * old_h_ratio, w_ratio * old_w_ratio,
            h_ratio * old_h_ratio
        ]
        results['crop_quadruple'] = np.array(
            new_crop_quadruple, dtype=np.float32)

        crop_bbox = np.array([left, top, right, bottom])
        results['crop_bbox'] = crop_bbox
        results['img_shape'] = (new_h, new_w)

        img_input_key = f'imgs_{self.input_key}' if self.input_key != '' else 'imgs'
        img_out_key = f'imgs_{self.out_key}' if self.out_key != '' else 'imgs'
        results[img_out_key] = self._crop_imgs(results[img_input_key], crop_bbox)

        if self.enable_futr:
            futr_img_input_key = f'future_imgs_{self.input_key}' if self.input_key != '' else 'future_imgs'
            futr_img_out_key = f'future_imgs_{self.out_key}' if self.out_key != '' else 'future_imgs'
            results[futr_img_out_key] = self._crop_imgs(results[futr_img_input_key], crop_bbox)

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'area_range={self.area_range}, '
                    f'aspect_ratio_range={self.aspect_ratio_range}, '
                    f'lazy={self.lazy})')
        return repr_str

@TRANSFORMS.register_module()
class ColorJitterCurFutrFrame(BaseTransform):
    """Perform ColorJitter to each img.

    Required keys are "imgs", added or modified keys are "imgs".

    Args:
        brightness (float | tuple[float]): The jitter range for brightness, if
            set as a float, the range will be (1 - brightness, 1 + brightness).
            Default: 0.5.
        contrast (float | tuple[float]): The jitter range for contrast, if set
            as a float, the range will be (1 - contrast, 1 + contrast).
            Default: 0.5.
        saturation (float | tuple[float]): The jitter range for saturation, if
            set as a float, the range will be (1 - saturation, 1 + saturation).
            Default: 0.5.
        hue (float | tuple[float]): The jitter range for hue, if set as a
            float, the range will be (-hue, hue). Default: 0.1.
    """

    @staticmethod
    def check_input(val, max, base):
        if isinstance(val, tuple):
            assert base - max <= val[0] <= val[1] <= base + max
            return val
        assert val <= max
        return (base - val, base + val)

    @staticmethod
    def rgb_to_grayscale(img):
        return 0.2989 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]

    @staticmethod
    def adjust_contrast(img, factor):
        val = np.mean(ColorJitterCurFutrFrame.rgb_to_grayscale(img))
        return factor * img + (1 - factor) * val

    @staticmethod
    def adjust_saturation(img, factor):
        gray = np.stack([ColorJitterCurFutrFrame.rgb_to_grayscale(img)] * 3, axis=-1)
        return factor * img + (1 - factor) * gray

    @staticmethod
    def adjust_hue(img, factor):
        img = np.clip(img, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        offset = int(factor * 255)
        hsv[..., 0] = (hsv[..., 0] + offset) % 180
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return img.astype(np.float32)

    def __init__(self, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, input_key='s', out_key='s', enable_futr=True):
        self.brightness = self.check_input(brightness, 1, 1)
        self.contrast = self.check_input(contrast, 1, 1)
        self.saturation = self.check_input(saturation, 1, 1)
        self.hue = self.check_input(hue, 0.5, 0)
        self.fn_idx = np.random.permutation(4)
        self.input_key = input_key
        self.out_key = out_key
        self.enable_futr = enable_futr

    def transform(self, results):
        """Perform ColorJitter.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        img_key = f'imgs_{self.input_key}' if self.input_key != '' else 'imgs'
        imgs = results[img_key]
        num_clips, clip_len = 1, len(imgs)

        new_imgs = []
        for i in range(num_clips):
            b = np.random.uniform(
                low=self.brightness[0], high=self.brightness[1])
            c = np.random.uniform(low=self.contrast[0], high=self.contrast[1])
            s = np.random.uniform(
                low=self.saturation[0], high=self.saturation[1])
            h = np.random.uniform(low=self.hue[0], high=self.hue[1])
            start, end = i * clip_len, (i + 1) * clip_len

            for img in imgs[start:end]:
                img = img.astype(np.float32)
                for fn_id in self.fn_idx:
                    if fn_id == 0 and b != 1:
                        img *= b
                    if fn_id == 1 and c != 1:
                        img = self.adjust_contrast(img, c)
                    if fn_id == 2 and s != 1:
                        img = self.adjust_saturation(img, s)
                    if fn_id == 3 and h != 0:
                        img = self.adjust_hue(img, h)
                img = np.clip(img, 0, 255).astype(np.uint8)
                new_imgs.append(img)
            results[f'imgs_{self.out_key}'] = new_imgs
            
            if self.enable_futr:
                futr_img_key = f'future_imgs_{self.input_key}' if self.input_key != '' else 'future_imgs'
                futr_imgs = results[futr_img_key]
                new_futr_imgs = []
                for img in futr_imgs:
                    img = img.astype(np.float32)
                    for fn_id in self.fn_idx:
                        if fn_id == 0 and b != 1:
                            img *= b
                        if fn_id == 1 and c != 1:
                            img = self.adjust_contrast(img, c)
                        if fn_id == 2 and s != 1:
                            img = self.adjust_saturation(img, s)
                        if fn_id == 3 and h != 0:
                            img = self.adjust_hue(img, h)
                    img = np.clip(img, 0, 255).astype(np.uint8)
                    new_futr_imgs.append(img)
                results[f'future_imgs_{self.out_key}'] = new_futr_imgs
        
        return results

@TRANSFORMS.register_module()
class RandomErasingCurFutrFrame(BaseTransform):
    """Randomly selects a rectangle region in an image and erase pixels.
    basically refer mmcls.

    **Required Keys:**

    - img

    **Modified Keys:**

    - img

    Args:
        erase_prob (float): Probability that image will be randomly erased.
            Default: 0.5
        min_area_ratio (float): Minimum erased area / input image area
            Default: 0.02
        max_area_ratio (float): Maximum erased area / input image area
            Default: 1/3
        aspect_range (sequence | float): Aspect ratio range of erased area.
            if float, it will be converted to (aspect_ratio, 1/aspect_ratio)
            Default: (3/10, 10/3)
        mode (str): Fill method in erased area, can be:

            - const (default): All pixels are assign with the same value.
            - rand: each pixel is assigned with a random value in [0, 255]

        fill_color (sequence | Number): Base color filled in erased area.
            Defaults to (128, 128, 128).
        fill_std (sequence | Number, optional): If set and ``mode`` is 'rand',
            fill erased area with random color from normal distribution
            (mean=fill_color, std=fill_std); If not set, fill erased area with
            random color from uniform distribution (0~255). Defaults to None.

    Note:
        See `Random Erasing Data Augmentation
        <https://arxiv.org/pdf/1708.04896.pdf>`_

        This paper provided 4 modes: RE-R, RE-M, RE-0, RE-255, and use RE-M as
        default. The config of these 4 modes are:

        - RE-R: RandomErasing(mode='rand')
        - RE-M: RandomErasing(mode='const', fill_color=(123.67, 116.3, 103.5))
        - RE-0: RandomErasing(mode='const', fill_color=0)
        - RE-255: RandomErasing(mode='const', fill_color=255)
    """

    def __init__(self,
                 erase_prob=0.5,
                 min_area_ratio=0.02,
                 max_area_ratio=1 / 3,
                 aspect_range=(3 / 10, 10 / 3),
                 mode='const',
                 fill_color=(128, 128, 128),
                 fill_std=None,
                 input_key='s',
                 out_key='s'):
        assert isinstance(erase_prob, float) and 0. <= erase_prob <= 1.
        assert isinstance(min_area_ratio, float) and 0. <= min_area_ratio <= 1.
        assert isinstance(max_area_ratio, float) and 0. <= max_area_ratio <= 1.
        assert min_area_ratio <= max_area_ratio, \
            'min_area_ratio should be smaller than max_area_ratio'
        if isinstance(aspect_range, float):
            aspect_range = min(aspect_range, 1 / aspect_range)
            aspect_range = (aspect_range, 1 / aspect_range)
        assert isinstance(aspect_range, Sequence) and len(aspect_range) == 2 \
            and all(isinstance(x, float) for x in aspect_range), \
            'aspect_range should be a float or Sequence with two float.'
        assert all(x > 0 for x in aspect_range), \
            'aspect_range should be positive.'
        assert aspect_range[0] <= aspect_range[1], \
            'In aspect_range (min, max), min should be smaller than max.'
        assert mode in ['const', 'rand'], \
            'Please select `mode` from ["const", "rand"].'
        if isinstance(fill_color, Number):
            fill_color = [fill_color] * 3
        assert isinstance(fill_color, Sequence) and len(fill_color) == 3 \
            and all(isinstance(x, Number) for x in fill_color), \
            'fill_color should be a float or Sequence with three int.'
        if fill_std is not None:
            if isinstance(fill_std, Number):
                fill_std = [fill_std] * 3
            assert isinstance(fill_std, Sequence) and len(fill_std) == 3 \
                and all(isinstance(x, Number) for x in fill_std), \
                'fill_std should be a float or Sequence with three int.'

        self.erase_prob = erase_prob
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.aspect_range = aspect_range
        self.mode = mode
        self.fill_color = fill_color
        self.fill_std = fill_std
        self.input_key = input_key
        self.out_key = out_key

    def _img_fill_pixels(self, img, top, left, h, w):
        """Fill pixels to the patch of image."""
        if self.mode == 'const':
            patch = np.empty((h, w, 3), dtype=np.uint8)
            patch[:, :] = np.array(self.fill_color, dtype=np.uint8)
        elif self.fill_std is None:
            # Uniform distribution
            patch = np.random.uniform(0, 256, (h, w, 3)).astype(np.uint8)
        else:
            # Normal distribution
            patch = np.random.normal(self.fill_color, self.fill_std, (h, w, 3))
            patch = np.clip(patch.astype(np.int32), 0, 255).astype(np.uint8)

        img[top:top + h, left:left + w] = patch
        return img

    def _fill_pixels(self, imgs, top, left, h, w):
        """Fill pixels to the patch of each image in frame clip."""
        return [self._img_fill_pixels(img, top, left, h, w) for img in imgs]

    @cache_randomness
    def random_disable(self):
        """Randomly disable the transform."""
        return np.random.rand() > self.erase_prob

    @cache_randomness
    def random_patch(self, img_h, img_w):
        """Randomly generate patch the erase."""
        # convert the aspect ratio to log space to equally handle width and
        # height.
        log_aspect_range = np.log(
            np.array(self.aspect_range, dtype=np.float32))
        aspect_ratio = np.exp(np.random.uniform(*log_aspect_range))
        area = img_h * img_w
        area *= np.random.uniform(self.min_area_ratio, self.max_area_ratio)

        h = min(int(round(np.sqrt(area * aspect_ratio))), img_h)
        w = min(int(round(np.sqrt(area / aspect_ratio))), img_w)
        top = np.random.randint(0, img_h - h) if img_h > h else 0
        left = np.random.randint(0, img_w - w) if img_w > w else 0
        return top, left, h, w

    def transform(self, results):
        """
        Args:
            results (dict): Results dict from pipeline

        Returns:
            dict: Results after the transformation.
        """
        if self.random_disable():
            return results

        img_input_key = f'imgs_{self.input_key}' if self.input_key != '' else 'imgs'
        img_out_key = f'imgs_{self.out_key}' if self.out_key != '' else 'imgs'
        futr_img_input_key = f'future_imgs_{self.input_key}' if self.input_key != '' else 'future_imgs'
        futr_img_out_key = f'future_imgs_{self.out_key}' if self.out_key != '' else 'future_imgs'

        imgs = results[img_input_key]
        img_h, img_w = imgs[0].shape[:2]
        imgs = self._fill_pixels(imgs, *self.random_patch(img_h, img_w))
        results[img_out_key] = imgs

        futr_imgs = results[futr_img_input_key]
        futr_imgs = self._fill_pixels(futr_imgs, *self.random_patch(img_h, img_w))
        results[futr_img_out_key] = futr_imgs
        return results

@TRANSFORMS.register_module()
class PytorchVideoWrapperCurFutrFrame(BaseTransform):
    """PytorchVideoTrans Augmentations, under pytorchvideo.transforms.

    Args:
        op (str): The name of the pytorchvideo transformation.
    """

    def __init__(self, op, 
                 input_key='s', 
                 out_key='s',
                 enable_futr=True,
                 **kwargs):
        try:
            import pytorchvideo.transforms as ptv_trans
            import torch
        except ImportError:
            raise RuntimeError('Install pytorchvideo to use PytorchVideoTrans')
        if digit_version(torch.__version__) < digit_version('1.8.0'):
            raise RuntimeError(
                'The version of PyTorch should be at least 1.8.0')

        trans = getattr(ptv_trans, op, None)
        assert trans, f'Transform {op} not in pytorchvideo'

        supported_pytorchvideo_trans = ('AugMix', 'RandAugment',
                                        'RandomResizedCrop', 'ShortSideScale',
                                        'RandomShortSideScale')
        assert op in supported_pytorchvideo_trans,\
            f'PytorchVideo Transform {op} is not supported in MMAction2'

        self.trans = trans(**kwargs)
        self.op = op
        self.input_key = input_key
        self.out_key   = out_key
        self.enable_futr = enable_futr

    def transform(self, results):
        """Perform PytorchVideoTrans augmentations.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        assert 'imgs' in results

        assert 'gt_bboxes' not in results,\
            f'PytorchVideo {self.op} doesn\'t support bboxes yet.'
        assert 'proposals' not in results,\
            f'PytorchVideo {self.op} doesn\'t support bboxes yet.'

        img_input_key = f'imgs_{self.input_key}' if self.input_key != '' else 'imgs'
        img_out_key = f'imgs_{self.out_key}' if self.out_key != '' else 'imgs'
        if self.enable_futr:
            futr_img_input_key = f'future_imgs_{self.input_key}' if self.input_key != '' else 'future_imgs'
            futr_img_out_key = f'future_imgs_{self.out_key}' if self.out_key != '' else 'future_imgs'

        if self.op in ('AugMix', 'RandAugment'):
            # list[ndarray(h, w, 3)] -> torch.tensor(t, c, h, w)
            imgs = [x.transpose(2, 0, 1) for x in results[img_input_key]]
            imgs = to_tensor(np.stack(imgs))

            if self.enable_futr:
                futr_imgs = [x.transpose(2, 0, 1) for x in results[futr_img_input_key]]
                futr_imgs = to_tensor(np.stack(futr_imgs))
        else:
            # list[ndarray(h, w, 3)] -> torch.tensor(c, t, h, w)
            # uint8 -> float32
            imgs = to_tensor((np.stack(results['imgs']).transpose(3, 0, 1, 2) /
                              255.).astype(np.float32))

        imgs = self.trans(imgs).data.numpy()

        if self.enable_futr:
            futr_imgs = self.trans(futr_imgs).data.numpy()

        if self.op in ('AugMix', 'RandAugment'):
            imgs[imgs > 255] = 255
            imgs[imgs < 0] = 0
            imgs = imgs.astype(np.uint8)

            # torch.tensor(t, c, h, w) -> list[ndarray(h, w, 3)]
            imgs = [x.transpose(1, 2, 0) for x in imgs]

            if self.enable_futr:
                futr_imgs[futr_imgs > 255] = 255
                futr_imgs[futr_imgs < 0] = 0
                futr_imgs = futr_imgs.astype(np.uint8)
                futr_imgs = [x.transpose(1, 2, 0) for x in futr_imgs]
        else:
            # float32 -> uint8
            imgs = imgs * 255
            imgs[imgs > 255] = 255
            imgs[imgs < 0] = 0
            imgs = imgs.astype(np.uint8)

            # torch.tensor(c, t, h, w) -> list[ndarray(h, w, 3)]
            imgs = [x for x in imgs.transpose(1, 2, 3, 0)]

        results[img_out_key] = imgs

        if self.enable_futr:
            results[futr_img_out_key] = futr_imgs

        return results

@TRANSFORMS.register_module()
class TeacherTransformsFrame(BaseTransform):
    def __init__(self, 
                 transforms: List[Dict[str, Union[str, float]]],
                 out_scale=(224, 224)) -> None:
        # TODO: randomly select a transform from the list
        self.transforms = [TRANSFORMS.build(t) for t in transforms]

    def transform(self, results: Dict) -> Dict:
        for t in self.transforms:
            results = t(results)
        # print("In teacher tfms >", results['next_clip_filename'], results['filename'])

        return results

    def __repr__(self) -> str:
        repr_str = (f'{self.__class__.__name__}('
                    f'transforms={self.transforms})')
        return repr_str
    
@TRANSFORMS.register_module()
class StudentTransformsFrame(BaseTransform):
    def __init__(self, 
                 transforms: List[Dict[str, Union[str, float]]],
                 out_scale=(224, 224)) -> None:
        # TODO: randomly select a transform from the list
        self.transforms = [TRANSFORMS.build(t) for t in transforms]

    def transform(self, results: Dict) -> Dict:
        for t in self.transforms:
            results = t(results)
            
        # print("In student tfms >", results['next_clip_filename'], results['filename'])
        # t_cur_clip = results['imgs_t'][0]
        # s_cur_clip = results['imgs_s'][0]
        # plt.imshow(t_cur_clip)
        # plt.show()
        # plt.imshow(s_cur_clip)
        # plt.show()

        return results

    def __repr__(self) -> str:
        repr_str = (f'{self.__class__.__name__}('
                    f'transforms={self.transforms})')
        return repr_str