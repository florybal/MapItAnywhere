from dataclasses import dataclass
from typing import Optional, Any, Dict, List
from pathlib import Path

@dataclass
class AugmentationConfiguration:
    gaussian_noise: dict
    brightness_contrast: dict

    enabled: bool = False
    brightness: float = 0.5
    contrast: float = 0.5
    saturation: float = 0.5
    hue: float = 0.5
    random_resized_crop: Any = False
    random_flip: float = 0.5


@dataclass(kw_only=True)
class DataConfiguration:
    augmentations: AugmentationConfiguration

    loading: Dict[str, Dict[str, Any]]

    target_focal_length: Optional[int] = None
    reduce_fov: Optional[bool] = None
    resize_image: Optional[Any] = None
    pad_to_square: Optional[bool] = None
    pad_to_multiple: Optional[int] = None
    gravity_align: Optional[bool] = None
    rectify_pitch: Optional[bool] = True
    num_classes: int

    name: str
    seed: Optional[int] = 0
    random: Optional[bool] = True
    num_threads: Optional[int] = None
    # for visualization/inference only: ignore FOV mask and render full square
    orthographic: bool = False

@dataclass(kw_only=True)
class MIADataConfiguration(DataConfiguration):

    scenes: list[str]
    split: Any
    data_dir: Path
    pixel_per_meter: int
    crop_size_meters: int

    name: str = "mapillary"
    filter_for: Optional[str] = None
    filter_by_ground_angle: Optional[float] = None
    min_num_points: int = 0

@dataclass(kw_only=True)
class KITTIDataConfiguration(DataConfiguration):
    seam_root_dir: Path
    dataset_root_dir: Path
    bev_percentage: float

    pixel_per_meter: int
    crop_size_meters: int

    class_mapping: Optional[Any] = None
    percentage: float = 1.0

@dataclass(kw_only=True)
class NuScenesDataConfiguration(DataConfiguration):
    data_dir: Path
    map_dir: Path
    pixel_per_meter: int
    crop_size_meters: int

    percentage: float = 1.0
    class_mapping: Optional[Any] = None
    version: str = "v1.0-trainval"


@dataclass(kw_only=True)
class IndoorDataConfiguration(DataConfiguration):
    """
    Configuração para dataset Indoor BEV (warehouse/industrial).
    
    Args:
        data_dir: Diretório raiz do dataset
        pixel_per_meter: Resolução do grid BEV (pixels por metro)
        crop_size_meters: Tamanho da área BEV em metros
        room_types: Lista de tipos de ambientes para filtrar (opcional)
        floor_levels: Lista de andares para filtrar (opcional)
        class_names: Lista com nomes das classes
        class_mapping: Mapeamento de classes (opcional)
    """
    
    # Diretórios
    data_dir: Path
    
    # Parâmetros de grid BEV
    pixel_per_meter: int
    crop_size_meters: int
    
    # Filtros opcionais
    room_types: Optional[List[str]] = None
    floor_levels: Optional[List[int]] = None
    
    # Classes
    class_names: Optional[List[str]] = None
    class_mapping: Optional[Any] = None
    
    # Defaults
    name: str = "indoor"
    percentage: float = 1.0  # Fração do dataset a usar
    