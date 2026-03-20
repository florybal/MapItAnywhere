import os
from matplotlib import pyplot as plt
from mapper.utils.io import read_image
from mapper.utils.exif import EXIF
from mapper.utils.wrappers import Camera
from mapper.data.image import rectify_image, resize_image
from mapper.utils.viz_2d import one_hot_argmax_to_rgb, plot_images
from mapper.module import GenericModule
# from perspective2d import PerspectiveFields  # removed dependency, using fixed calibration
import torch
import numpy as np
from typing import Optional, Tuple
import glob 
import hydra
from hydra.core.config_store import ConfigStore
from typing import Any
from dataclasses import dataclass

from .models.schema import ModelConfiguration, DINOConfiguration, ResNetConfiguration
from .data.schema import MIADataConfiguration, KITTIDataConfiguration, NuScenesDataConfiguration

@dataclass
class ExperimentConfiguration:
    name: str

@dataclass
class Configuration:
    model: ModelConfiguration
    experiment: ExperimentConfiguration
    data: Any
    training: Any

    image_path: str
    save_path: str = "output.png"


cs = ConfigStore.instance()

# Store root configuration schema
cs.store(name="pretrain", node=Configuration)
cs.store(name="mapper_nuscenes", node=Configuration)
cs.store(name="mapper_kitti", node=Configuration)

# Store data configuration schema
cs.store(group="schema/data", name="mia",
         node=MIADataConfiguration, package="data")
cs.store(group="schema/data", name="kitti", node=KITTIDataConfiguration, package="data")
cs.store(group="schema/data", name="nuscenes", node=NuScenesDataConfiguration, package="data")

cs.store(group="model/schema/backbone", name="dino", node=DINOConfiguration, package="model.image_encoder.backbone")
cs.store(group="model/schema/backbone", name="resnet", node=ResNetConfiguration, package="model.image_encoder.backbone")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# simplified calibrator that returns zero roll/pitch and default camera parameters
class ImageCalibrator:
    def __init__(self, version: str = "stub"):
        pass

    def to(self, device):
        # compatibility method
        return self

    def run(
        self,
        image_rgb: np.ndarray,
        focal_length: Optional[float] = None,
        exif: Optional[EXIF] = None,
    ) -> Tuple[Tuple[float, float], Camera]:
        # no calibration, assume camera center and no rotation
        h, w, *_ = image_rgb.shape
        if focal_length is None:
            focal_length = max(w, h)
        camera = Camera.from_dict(
            {
                "model": "SIMPLE_PINHOLE",
                "width": w,
                "height": h,
                "params": [focal_length, w / 2 + 0.5, h / 2 + 0.5],
            }
        )
        roll_pitch = (0.0, 0.0)
        return roll_pitch, camera

def preprocess_pipeline(image, roll_pitch, camera):
    image = torch.from_numpy(image).float() / 255
    image = image.permute(2, 0, 1).to(device)
    camera = camera.to(device)

    image, valid = rectify_image(image, camera.float(), -roll_pitch[0], -roll_pitch[1])

    roll_pitch *= 0

    image, _, camera, valid = resize_image(
        image=image,
        size=512,
        camera=camera,
        fn=max,
        valid=valid
    )

    camera = torch.stack([camera])

    return {
        "image": image.unsqueeze(0).to(device),
        "valid": valid.unsqueeze(0).to(device),
        "camera": camera.float().to(device),
    }


def infer(calibrator, model, image_path: str, cfg):

    image = read_image(image_path)
    with open(image_path, "rb") as fid:
        exif = EXIF(fid, lambda: image.shape[:2])
    
    gravity, camera = calibrator.run(image, exif=exif)

    data = preprocess_pipeline(image, gravity, camera)
    res = model(data)
    
    prediction = res['output']
    # number of classes comes from configuration (data or model)
    ncls = cfg.data.num_classes if hasattr(cfg.data, 'num_classes') else cfg.model.num_classes

    # try to load color map and labels from dataset directory if present
    color_map = None
    legend_labels = None
    data_dir = getattr(cfg.data, 'data_dir', None)
    if data_dir is not None:
        color_file = os.path.join(os.path.abspath(str(data_dir)), 'label_colors.txt')
        if os.path.exists(color_file):
            from mapper.utils.viz_2d import load_label_colors
            colors_list, legend_labels = load_label_colors(color_file)
            print("Loaded legend labels from data_dir:", legend_labels)
            # if first label is background, force grey and optionally hide
            if legend_labels and legend_labels[0].lower() == 'background':
                colors_list[0] = (128,128,128)
                # optionally mark it as "background (ignored)" or remove from legend
                legend_labels[0] = 'background'
            # ensure list length matches number of classes + void
            if len(colors_list) < ncls + 1:
                colors_list = colors_list + [None] * (ncls + 1 - len(colors_list))
            color_map = colors_list
            # attach legend info for plot_images (will reference global import)
            plot_images._legend_info = (legend_labels, colors_list)

    # if color_map still None, try to locate label_colors.txt walking up from image folder
    if color_map is None:
        img_dir = os.path.dirname(image_path)
        cur = img_dir
        while True:
            candidate = os.path.join(cur, 'label_colors.txt')
            if os.path.exists(candidate):
                from mapper.utils.viz_2d import load_label_colors
                colors_list, legend_labels = load_label_colors(candidate)
                print("Loaded legend labels from image path:", legend_labels)
                if legend_labels and legend_labels[0].lower() == 'background':
                    colors_list[0] = (128,128,128)
                    legend_labels[0] = 'background'
                if len(colors_list) < ncls + 1:
                    colors_list = colors_list + [None] * (ncls + 1 - len(colors_list))
                color_map = colors_list
                plot_images._legend_info = (legend_labels, colors_list)
                break
            parent = os.path.dirname(cur)
            if parent == cur or parent == '':
                break
            cur = parent

    # final fallback: use class_names from config if available
    if plot_images._legend_info is None and hasattr(cfg.data, 'class_names'):
        names = list(cfg.data.class_names)
        # ensure background entry if missing
        if len(names) < ncls + 1:
            names = names + [f'class_{i}' for i in range(len(names), ncls + 1)]
        print("Using class_names from config for legend:", names)
        # color_map may still be None; leave it to default palette
        plot_images._legend_info = (names, color_map if color_map is not None else [])

    rgb_prediction = one_hot_argmax_to_rgb(prediction, ncls, class_colors=color_map).squeeze(0).permute(1, 2, 0).cpu().long().numpy()
    valid = res['valid_bev'].squeeze(0)[..., :-1]
    # if user requests orthographic output (e.g. top-down camera), do not mask
    orth = getattr(cfg.data, 'orthographic', False)
    if not orth:
        rgb_prediction[~valid.cpu().numpy()] = 255
    
    plot_images([image, rgb_prediction], titles=["Input Image", "Top-Down Prediction"], pad=2, adaptive=True)

    # clear legend info so subsequent calls aren't contaminated
    plot_images._legend_info = None

    return plt.gcf()

@hydra.main(version_base=None, config_path="conf", config_name="pretrain")
def main(cfg: Configuration):
    calibrator = ImageCalibrator().to(device)

    model = GenericModule(cfg)
    state_dict = torch.load(cfg.training.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state_dict["state_dict"], strict=False)
    model = model.to(device)
    model = model.eval()

    fig = infer(calibrator, model, cfg.image_path, cfg)
    
    # Determine where to write the output.  The user may pass an absolute
    # path, but we force the file to live under the current working directory
    # (i.e. the project root) so that results are always visible inside the
    # repo.  If an absolute path is given we take its basename and place it in
    # cwd/.
    save_path = cfg.save_path
    if not save_path.endswith('.png'):
        save_path = save_path + '.png'

    # if the path is absolute or points outside cwd, rewrite to cwd/<basename>
    if os.path.isabs(save_path) or not os.path.commonpath([os.getcwd(), os.path.abspath(save_path)]).startswith(os.getcwd()):
        save_path = os.path.join(os.getcwd(), os.path.basename(save_path))

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig.savefig(save_path)
    print(f"Output saved to: {save_path}")

if __name__ == "__main__":
    main()