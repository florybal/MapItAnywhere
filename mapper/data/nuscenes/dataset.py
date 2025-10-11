import os
import torch
import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from itertools import chain
from PIL import Image
from torchvision import transforms as T
import torchvision.transforms as tvf
from torchvision.transforms.functional import to_tensor

from .splits_roddick import create_splits_scenes_roddick
from ..image import pad_image, rectify_image, resize_image
from .utils import decode_binary_labels
from ..utils import decompose_rotmat
from ...utils.io import read_image
from ...utils.wrappers import Camera
from ..schema import NuScenesDataConfiguration
from nuscenes.map_expansion.map_api import NuScenesMap
import math


def _quat_to_yaw_deg(self, q):
    # q: [w, x, y, z] do nuScenes (lista) -> yaw em graus
    # nuScenes guarda como [w, x, y, z]; pyquaternion aceita nessa ordem.
    R = Quaternion(q).rotation_matrix
    yaw = math.atan2(R[1, 0], R[0, 0])  # z-rotation
    return math.degrees(yaw)

def _render_map_mask(self, location, center_xy, yaw_deg, canvas_hw=(800, 800),
                     patch_size_m=100.0, layer_names=None):
    """
    Rasteriza camadas do mapa (nuScenes devkit) para um stack binário [H, W, C].
    - center_xy: (x, y) global (metros)
    - yaw_deg: orientação do carro em graus (global->ego)
    - canvas_hw: (H, W) da saída
    - patch_size_m: lado do quadrado em metros
    - layer_names: lista de camadas semânticas do nuScenes
    """
    if layer_names is None:
        # escolha enxuta; ajuste conforme suas classes/treino
        layer_names = [
            "drivable_area",
            "ped_crossing",
            "walkway",
            "carpark_area",
            "road_segment",         # alternativa à 'drivable_area' p/ vias
            "lane"                  # opcional; remova/ajuste conforme classes
        ]
    H, W = canvas_hw
    nusc_map = NuScenesMap(self.nusc.dataroot, location=location)

    # patch_box: (cx, cy, width, height) em METROS no frame global
    patch_box = (center_xy[0], center_xy[1], patch_size_m, patch_size_m)

    # get_map_mask retorna (layers, H, W) ou (H, W, layers) dependendo da versão;
    # vamos padronizar para (H, W, C)
    mask = nusc_map.get_map_mask(
        patch_box=patch_box,
        patch_angle=yaw_deg,             # alinha com o heading do veículo
        layer_names=layer_names,
        canvas_size=(H, W)
    )
    # normalizar shape → (H, W, C)
    import numpy as np
    if isinstance(mask, list):  # versões antigas
        mask = np.stack(mask, axis=-1).astype(np.uint8)
    else:
        # algumas versões retornam (C, H, W)
        if mask.ndim == 3 and mask.shape[0] == len(layer_names):
            mask = np.transpose(mask, (1, 2, 0)).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)

    # clipe binário {0,1}
    mask = (mask > 0).astype(np.uint8)

    # canal de "visibility"/confiança: 1 onde há informação
    vis = (mask.sum(axis=-1, keepdims=True) > 0).astype(np.uint8)

    # Empilha [C] + [vis] → total C+1 (o código depois espera “+1” para flood/visibility)
    out = np.concatenate([mask, vis], axis=-1)
    return out  # (H, W, C+1), values {0,1}



class NuScenesDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: NuScenesDataConfiguration, split="train"):

        self.cfg = cfg
        self.nusc = NuScenes(version=cfg.version, dataroot=str(cfg.data_dir))
        self.map_data_root = cfg.map_dir
        self.split = split

        self.scenes = create_splits_scenes_roddick() # custom based on Roddick et al. 

        scene_split = {
            'v1.0-trainval': {'train': 'train', 'val': 'val', 'test': 'val'},
            'v1.0-mini': {'train': 'mini_train', 'val': 'mini_val'},
        }[cfg.version][split]
        self.scenes = self.scenes[scene_split]
        self.sample = list(filter(lambda sample: self.nusc.get(
            'scene', sample['scene_token'])['name'] in self.scenes, self.nusc.sample))

        self.tfs = self.get_augmentations() if split == "train" else T.Compose([])

        data_tokens = []
        for sample in self.sample:
            data_token = sample['data']
            data_token = [v for k,v in data_token.items() if k == "CAM_FRONT"]

            data_tokens.append(data_token)

        data_tokens = list(chain.from_iterable(data_tokens))
        data = [self.nusc.get('sample_data', token) for token in data_tokens]

        self.data = []
        for d in data:
            sample = self.nusc.get('sample', d['sample_token'])
            scene = self.nusc.get('scene', sample['scene_token'])
            location = self.nusc.get('log', scene['log_token'])['location']

            file_name = d['filename']
            ego_pose = self.nusc.get('ego_pose', d['ego_pose_token'])
            calibrated_sensor = self.nusc.get(
                "calibrated_sensor", d['calibrated_sensor_token'])

            ego2global = np.eye(4).astype(np.float32)
            ego2global[:3, :3] = Quaternion(ego_pose['rotation']).rotation_matrix
            ego2global[:3, 3] = ego_pose['translation']

            sensor2ego = np.eye(4).astype(np.float32)
            sensor2ego[:3, :3] = Quaternion(
                calibrated_sensor['rotation']).rotation_matrix
            sensor2ego[:3, 3] = calibrated_sensor['translation']

            sensor2global = ego2global @ sensor2ego

            rotation = sensor2global[:3, :3]
            roll, pitch, yaw = decompose_rotmat(rotation)

            fx = calibrated_sensor['camera_intrinsic'][0][0]
            fy = calibrated_sensor['camera_intrinsic'][1][1]
            cx = calibrated_sensor['camera_intrinsic'][0][2]
            cy = calibrated_sensor['camera_intrinsic'][1][2]
            width = d['width']
            height = d['height']

            cam = Camera(torch.tensor(
                [width, height, fx, fy, cx - 0.5, cy - 0.5])).float()
            self.data.append({
                'filename': file_name,
                'yaw': yaw,
                'pitch': pitch,
                'roll': roll,
                'cam': cam,
                'sensor2global': sensor2global,
                'token': d['token'],
                'sample_token': d['sample_token'],
                'location': location
            })
        
        if self.cfg.percentage < 1.0 and split == "train":
            self.data = self.data[:int(len(self.data) * self.cfg.percentage)]

    def get_augmentations(self):

        print(f"Augmentation!", "\n" * 10)
        augmentations = [
            tvf.ColorJitter(
                brightness=self.cfg.augmentations.brightness,
                contrast=self.cfg.augmentations.contrast,
                saturation=self.cfg.augmentations.saturation,
                hue=self.cfg.augmentations.hue,
            )
        ]

        if self.cfg.augmentations.random_resized_crop:
            augmentations.append(
                tvf.RandomResizedCrop(scale=(0.8, 1.0))
            )  # RandomResizedCrop

        if self.cfg.augmentations.gaussian_noise.enabled:
            augmentations.append(
                tvf.GaussianNoise(
                    mean=self.cfg.augmentations.gaussian_noise.mean,
                    std=self.cfg.augmentations.gaussian_noise.std,
                )
            )  # Gaussian noise

        if self.cfg.augmentations.brightness_contrast.enabled:
            augmentations.append(
                tvf.ColorJitter(
                    brightness=self.cfg.augmentations.brightness_contrast.brightness_factor,
                    contrast=self.cfg.augmentations.brightness_contrast.contrast_factor,
                    saturation=0,  # Keep saturation at 0 for brightness and contrast adjustment
                    hue=0,
                )
            )  # Brightness and contrast adjustment

        return tvf.Compose(augmentations)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]

        image = read_image(os.path.join(self.nusc.dataroot, d['filename']))
        image = np.array(image)
        cam = d['cam']
        roll = d['roll']
        pitch = d['pitch']
        yaw = d['yaw']
        
        p = self.map_data_root / f"{d['token']}.png"
    
        if p.exists():
           with Image.open(p) as semantic_image:
               semantic_mask = to_tensor(semantic_image)
        else:
           # Fallback: gera máscara on-the-fly com nuScenes devkit
           # escolha um canvas compatível com sua head (ajuste se quiser)
           canvas_hw = (800, 800)  # (H, W)
           yaw_deg = self._quat_to_yaw_deg(self.nusc.get('ego_pose', self.nusc.get('sample_data', d['token'])['ego_pose_token'])['rotation'])
           m = self._render_map_mask(
               location=d['location'],
               center_xy=(d['sensor2global'][0, 3], d['sensor2global'][1, 3]),
               yaw_deg=yaw_deg,
               canvas_hw=canvas_hw,
               patch_size_m=100.0,    # ajuste se quiser mais/menos alcance
               layer_names=None       # ou passe sua lista
           )
           # (H, W, C+1) → (C+1, H, W) como to_tensor
           semantic_mask = torch.from_numpy(np.ascontiguousarray(m)).permute(2, 0, 1).float()

        semantic_mask = decode_binary_labels(semantic_mask, self.cfg.num_classes + 1)
        semantic_mask = torch.nn.functional.max_pool2d(semantic_mask.float(), (2, 2), stride=2) # 2 times downsample
        semantic_mask = semantic_mask.permute(1, 2, 0)
        semantic_mask = torch.flip(semantic_mask, [0])
        
        visibility_mask = semantic_mask[..., -1]
        semantic_mask = semantic_mask[..., :-1]

        if self.cfg.class_mapping is not None:
            semantic_mask = semantic_mask[..., self.cfg.class_mapping]

        image = (
            torch.from_numpy(np.ascontiguousarray(image))
            .permute(2, 0, 1)
            .float()
            .div_(255)
        )

        if not self.cfg.gravity_align:
            # Turn off gravity alignment
            roll = 0.0
            pitch = 0.0
            image, valid = rectify_image(image, cam, roll, pitch)

        else:
            image, valid = rectify_image(
                image, cam, roll, pitch if self.cfg.rectify_pitch else None
            )
            roll = 0.0
            if self.cfg.rectify_pitch:
                pitch = 0.0
        if self.cfg.resize_image is not None:
            image, _, cam, valid = resize_image(
                image, self.cfg.resize_image, fn=max, camera=cam, valid=valid
            )
            if self.cfg.pad_to_square:
                image, valid, cam = pad_image(image, self.cfg.resize_image, cam, valid)
        image = self.tfs(image)

        confidence_map = visibility_mask.clone().float()
        confidence_map = (confidence_map - confidence_map.min()) / (confidence_map.max() - confidence_map.min())

        return {
            "image": image,
            "roll_pitch_yaw": torch.tensor([roll, pitch, yaw]).float(),
            "camera": cam,
            "valid": valid,
            "seg_masks": semantic_mask.float(),
            "token": d['token'],
            "sample_token": d['sample_token'],
            'location': d['location'],
            'flood_masks': visibility_mask.float(),
            "confidence_map": confidence_map,
            'name': d['sample_token']
        }

