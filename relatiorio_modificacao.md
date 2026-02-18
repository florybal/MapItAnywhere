# RELATÓRIO TÉCNICO: Adaptação do Modelo MIA para Dataset Indoor BEV

**Projeto:** Map It Anywhere (MIA) - Adaptação para Ambientes Indoor  
**Data:** 12 de fevereiro de 2026  
**Objetivo:** Documentar todas as modificações necessárias para integrar um dataset indoor ao modelo MIA existente

---

## SUMÁRIO EXECUTIVO

Este relatório documenta as modificações necessárias para adaptar o modelo **Map It Anywhere (MIA)** para trabalhar com um dataset de **Bird's Eye View (BEV) indoor**. O modelo atualmente suporta datasets outdoor (KITTI-360, nuScenes, Mapillary), e requer adaptações específicas em código e configuração para processar dados de ambientes internos.

**Escopo Total:**
- ✅ 3 novos arquivos Python (~300 linhas)
- ✅ 3 arquivos Python existentes a modificar (~25 linhas adicionais)
- ✅ 2 novos arquivos de configuração YAML (~80 linhas)
- ✅ Organização de dados e estrutura de diretórios

---

## 1. ANÁLISE DA ARQUITETURA ATUAL

### 1.1 Pipeline do Modelo MIA

O modelo MIA executa o seguinte pipeline para gerar mapas BEV:

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Imagem     │────>│    Image     │────>│   Projeção   │────>│ Segmentação  │
│     FPV      │     │   Encoder    │     │     BEV      │     │   Semântica  │
│  (RGB, HxW)  │     │ (DINOv2/RN)  │     │ (Polar->Cart)│     │   (Classes)  │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

**Componentes principais:**

1. **Image Encoder**: Extrai features visuais da imagem
   - DINOv2 (padrão): output_dim=128, pretrained
   - ResNet: Alternativa configurável

2. **Scale Classifier**: Prediz profundidade/escala por pixel
   - Linear layer: latent_dim → num_scale_bins (32)
   - Permite projeção em múltiplas profundidades

3. **Polar Projection**: Projeta features em coordenadas polares
   - Utiliza depth steps de z_min a z_max
   - Resolução controlada por pixel_per_meter

4. **Cartesian Projection**: Converte coordenadas polares para BEV cartesiano
   - Grid final: (2×x_max, z_max) metros
   - Saída em formato (B, C, H, W)

5. **Segmentation Head**: Gera mapa semântico multi-classe
   - Decoder com dropout
   - Saída: (B, num_classes, H, W)

### 1.2 Datasets Atualmente Suportados

#### KITTI-360
- **Classes:** 8 (road, sidewalk, building, terrain, etc.)
- **Resolução:** 2 pixels/metro
- **Área BEV:** 50m profundidade × 50m largura
- **Formato:** Máscaras pré-computadas (.png) + metadados (.bin)
- **Características:** Máscaras de visibilidade, suporte a percentage training

#### nuScenes
- **Classes:** 14 (mapeadas para 6 em alguns configs)
- **Resolução:** 2 pixels/metro
- **Área BEV:** 50m × 50m
- **Formato:** API oficial + mapas rasterizados
- **Características:** Múltiplas câmeras, transformações ego2global

#### MIA (Mapillary)
- **Classes:** 6 (road, sidewalk, crosswalk, terrain, building, vehicle)
- **Resolução:** 2 pixels/metro
- **Área BEV:** 64m crop centralizado
- **Formato:** .npy arrays + flood masks
- **Características:** Confidence maps, splits por cidades

### 1.3 Estrutura de Dados Esperada

Todos os datasets retornam um dicionário com a seguinte estrutura:

```python
{
    'image': Tensor,              # (C, H, W) - Imagem RGB processada
    'camera': Camera,             # Objeto com parâmetros intrínsecos (fx, fy, cx, cy)
    'seg_masks': Tensor,          # (H, W, C) - Máscaras BEV one-hot encoded
    'flood_masks': Tensor,        # (H, W) - Máscara de visibilidade/validade
    'confidence_map': Tensor,     # (H, W) - 0=observável, 1=não-observável
    'roll_pitch_yaw': Tensor,     # (3,) - Orientação da câmera em radianos
    'valid': Tensor,              # (H, W) - Pixels válidos na imagem
    
    # Metadados opcionais
    'index': int,
    'name': str,
    'scene': str,
}
```

**Importante:** O modelo espera `seg_masks` no formato **(H, W, num_classes)** one-hot encoded, onde:
- H, W: Dimensões do grid BEV (z_max × 2×x_max em pixels)
- num_classes: Número de classes semânticas
- Valores: 0 ou 1 (binário) para cada classe

---

## 2. DIFERENÇAS ENTRE OUTDOOR E INDOOR

### 2.1 Comparação de Características

| Aspecto | Outdoor (Atual) | Indoor (Alvo) | Impacto |
|---------|----------------|---------------|---------|
| **Profundidade** | 50-100m | 10-20m | Reduzir `z_max` |
| **Largura Lateral** | 25-50m | 8-12m | Reduzir `x_max` |
| **Resolução** | 2 px/m | 4-10 px/m | Aumentar `pixel_per_meter` |
| **Classes** | Road-centric | Room-centric | Redefinir lista de classes |
| **Iluminação** | Natural, variável | Artificial, controlada | Ajustar augmentations |
| **Geometria** | Plana, road | Multi-nível, estruturado | Ajustar plane_params (opcional) |
| **Oclusões** | Veículos, edifícios | Móveis, portas, pessoas | Ajustar flood_masks |
| **FOV Típico** | ~90° horizontal | ~90-120° | Manter configuração |

### 2.2 Classes do Dataset Warehouse/Industrial

Classes definidas para o projeto de navegação autônoma em ambiente industrial:

```python
WAREHOUSE_CLASSES = [
    # Classe 0: Obstrução
    "obstrucao",            # Chapas de metal, lixeiras, cones, etc.
    
    # Classe 1: Empilhadeira
    "empilhadeira",         # Outras empilhadeiras/transpaleteiras ou parte do próprio robô
    
    # Classe 2: Carga
    "carga",                # Pallet, caixas de carregamento ou carrinhos de reboque
    
    # Classe 3: Máquinas
    "maquina",              # Todas máquinas ou esteiras industriais paradas no ambiente
    
    # Classe 4: Humano
    "humano",               # Humanos presentes no ambiente
    
    # Classe 5: Área Navegável
    "navegavel",            # Área livre para navegação (chão livre)
    
    # Classe 6: Estrutura
    "estrutura",            # Colunas, pilares, paredes e demais partes fixas
    
    # Classe 7: Porta-palete
    "portapalete",          # Estrutura de porta-palete (racks de armazenamento)
]
```

**Total:** 8 classes

**Cores associadas (para visualização):**

| Classe | Nome | Cor Hex | Nome da Cor |
|--------|------|---------|-------------|
| 0 | obstrucao | #FF0000 | Vermelho |
| 1 | empilhadeira | #FF8C00 | Laranja escuro |
| 2 | carga | #8A2BE2 | Azul-violeta |
| 3 | maquina | #0000FF | Azul |
| 4 | humano | #00AAFF | Azul-celeste vívido |
| 5 | navegavel | #00FF00 | Verde-limão |
| 6 | estrutura | #FFFF00 | Amarelo |
| 7 | portapalete | #CD853F | Castanho Peru |

### 2.3 Parâmetros Recomendados

```yaml
# Parâmetros para ambiente warehouse/industrial (corredor 4-5m, visão 12-15m)
pixel_per_meter: 5          # Alta resolução (~20cm/pixel)
crop_size_meters: 15        # Área observável
z_max: 15                   # Profundidade máxima (15m de visão frontal)
x_max: 6                    # Largura lateral (6m cada lado = 12m total - cobertura de corredor)
num_scale_bins: 24          # Menos bins que outdoor (adaptado para indoor)
```

---

## 3. MODIFICAÇÕES NECESSÁRIAS

### 3.1 Visão Geral das Modificações

```
mapper/
├── data/
│   ├── indoor/                    # ✅ CRIAR - Novo módulo indoor
│   │   ├── __init__.py           # ✅ CRIAR - Arquivo vazio
│   │   ├── dataset.py            # ✅ CRIAR - Dataset principal (~200 linhas)
│   │   └── data_module.py        # ✅ CRIAR - DataModule Lightning (~80 linhas)
│   │
│   ├── schema.py                  # ⚠️ MODIFICAR - Adicionar IndoorDataConfiguration
│   └── module.py                  # ⚠️ MODIFICAR - Registrar dataset indoor
│
├── conf/
│   ├── data/
│   │   └── indoor.yaml           # ✅ CRIAR - Config do dataset (~50 linhas)
│   │
│   └── mapper_indoor.yaml        # ✅ CRIAR - Config principal (~30 linhas)
│
└── mapper.py                      # ⚠️ MODIFICAR - Registrar schema no ConfigStore
```

**Legenda:**
- ✅ **CRIAR**: Arquivo novo a ser criado
- ⚠️ **MODIFICAR**: Arquivo existente a ser modificado

---

## 4. IMPLEMENTAÇÃO DETALHADA

### 4.1 Dataset Module - `mapper/data/indoor/dataset.py`

**Localização:** `mapper/data/indoor/dataset.py`  
**Tipo:** Novo arquivo  
**Linhas:** ~200-250  
**Descrição:** Implementa a classe principal do dataset que carrega imagens e anotações

**Estrutura:**

```python
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import json
import torchvision.transforms as tvf
from typing import Optional

from ..schema import IndoorDataConfiguration
from ..image import pad_image, rectify_image, resize_image
from ...utils.wrappers import Camera
from ...utils.io import read_image
from ..utils import decompose_rotmat


class IndoorBEVDataset(torch.utils.data.Dataset):
    """
    Dataset para ambientes indoor com anotações BEV.
    
    Estrutura esperada de diretórios:
        data_dir/
            images/
                scene1_frame001.jpg
                scene1_frame002.jpg
            annotations/
                scene1_frame001.npy  # Shape: (H, W, num_classes)
            camera_params/
                scene1_frame001.json
            flood_masks/  # Opcional
                scene1_frame001.npy
    """
    
    def __init__(self, cfg: IndoorDataConfiguration, split="train"):
        super().__init__()
        self.cfg = cfg
        self.split = split
        
        # Diretórios
        self.data_dir = Path(cfg.data_dir)
        self.image_dir = self.data_dir / "images"
        self.annotation_dir = self.data_dir / "annotations"
        self.camera_dir = self.data_dir / "camera_params"
        self.flood_mask_dir = self.data_dir / "flood_masks"
        
        # Carregar lista de frames do split
        split_file = self.data_dir / "splits" / f"{split}.txt"
        with open(split_file, 'r') as f:
            self.frame_names = [line.strip() for line in f.readlines()]
        
        # Filtros opcionais
        if hasattr(cfg, 'room_types') and cfg.room_types:
            self.frame_names = [
                f for f in self.frame_names 
                if any(room in f for room in cfg.room_types)
            ]
        
        # Augmentations
        self.augmentations = self.get_augmentations()
        
        print(f"[Indoor Dataset] Loaded {len(self.frame_names)} frames for {split}")
    
    def __len__(self):
        return len(self.frame_names)
    
    def __getitem__(self, idx):
        # Seed para reprodutibilidade
        if self.split == "train" and self.cfg.random:
            seed = None
        else:
            seed = [self.cfg.seed, idx]
        (seed,) = np.random.SeedSequence(seed).generate_state(1)
        
        frame_name = self.frame_names[idx]
        
        # Carregar imagem RGB
        image_path = self.image_dir / f"{frame_name}.jpg"
        image = read_image(str(image_path))
        image = Image.fromarray(image)
        image = self.augmentations(image)
        image = np.array(image)
        
        # Carregar parâmetros da câmera
        camera_path = self.camera_dir / f"{frame_name}.json"
        with open(camera_path, 'r') as f:
            cam_params = json.load(f)
        
        cam = Camera(torch.tensor([
            cam_params['width'],
            cam_params['height'],
            cam_params['fx'],
            cam_params['fy'],
            cam_params['cx'] - 0.5,
            cam_params['cy'] - 0.5,
        ])).float()
        
        roll = cam_params.get('roll', 0.0)
        pitch = cam_params.get('pitch', 0.0)
        yaw = cam_params.get('yaw', 0.0)
        
        # Processar imagem (resize, pad, rectify)
        image, valid, cam, roll, pitch = self.process_image(
            image, cam, roll, pitch, seed
        )
        
        # Carregar máscara de segmentação BEV
        annotation_path = self.annotation_dir / f"{frame_name}.npy"
        seg_masks = np.load(annotation_path)  # Shape: (H, W, num_classes)
        
        # Validar shape
        expected_classes = self.cfg.num_classes
        if seg_masks.shape[-1] != expected_classes:
            raise ValueError(
                f"Annotation {frame_name} has {seg_masks.shape[-1]} classes, "
                f"expected {expected_classes}"
            )
        
        seg_masks = torch.from_numpy(seg_masks).float()
        
        # Carregar flood mask (ou criar padrão)
        if self.flood_mask_dir.exists():
            flood_mask_path = self.flood_mask_dir / f"{frame_name}.npy"
            if flood_mask_path.exists():
                flood_mask = np.load(flood_mask_path)
            else:
                # Flood mask padrão: tudo observável (0)
                flood_mask = np.zeros(seg_masks.shape[:2], dtype=np.float32)
        else:
            flood_mask = np.zeros(seg_masks.shape[:2], dtype=np.float32)
        
        flood_mask = torch.from_numpy(flood_mask).float()
        
        # Confidence map: 0=observável, 1=não-observável
        confidence_map = flood_mask.clone()
        
        # Map Augmentations (flip horizontal)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            image, cam, valid, seg_masks, flood_mask, confidence_map = \
                self.random_flip(image, cam, valid, seg_masks, flood_mask, confidence_map)
        
        return {
            "index": idx,
            "name": frame_name,
            "scene": frame_name.split('_')[0],  # Primeiro token = scene
            "image": image,
            "valid": valid,
            "camera": cam,
            "seg_masks": seg_masks,
            "flood_masks": flood_mask,
            "roll_pitch_yaw": torch.tensor((roll, pitch, yaw)).float(),
            "confidence_map": confidence_map,
        }
    
    def get_augmentations(self):
        """Retorna transformações de augmentation para imagens."""
        if self.split != "train" or not self.cfg.augmentations.enabled:
            return tvf.Compose([])
        
        augmentations = [
            tvf.ColorJitter(
                brightness=self.cfg.augmentations.brightness,
                contrast=self.cfg.augmentations.contrast,
                saturation=self.cfg.augmentations.saturation,
                hue=self.cfg.augmentations.hue,
            )
        ]
        
        if self.cfg.augmentations.gaussian_noise.enabled:
            augmentations.append(
                tvf.GaussianNoise(
                    mean=self.cfg.augmentations.gaussian_noise.mean,
                    std=self.cfg.augmentations.gaussian_noise.std,
                )
            )
        
        if self.cfg.augmentations.brightness_contrast.enabled:
            augmentations.append(
                tvf.ColorJitter(
                    brightness=self.cfg.augmentations.brightness_contrast.brightness_factor,
                    contrast=self.cfg.augmentations.brightness_contrast.contrast_factor,
                    saturation=0,
                    hue=0,
                )
            )
        
        return tvf.Compose(augmentations)
    
    def random_flip(self, image, cam, valid, seg_mask, flood_mask, conf_mask):
        """Aplica flip horizontal aleatório."""
        if torch.rand(1) < self.cfg.augmentations.random_flip:
            image = torch.flip(image, [-1])
            cam = cam.flip()
            valid = torch.flip(valid, [-1])
            seg_mask = torch.flip(seg_mask, [1])  # Flip em X (largura)
            flood_mask = torch.flip(flood_mask, [-1])
            conf_mask = torch.flip(conf_mask, [-1])
        
        return image, cam, valid, seg_mask, flood_mask, conf_mask
    
    def process_image(self, image, cam, roll, pitch, seed):
        """
        Processa imagem: resize, pad, retificação de pitch.
        Adaptado de MapillaryDataset.
        """
        # Converter para tensor
        image = (
            torch.from_numpy(np.ascontiguousarray(image))
            .permute(2, 0, 1)
            .float()
            / 255.0
        )
        
        # Resize se configurado
        if self.cfg.resize_image is not None:
            image, cam = resize_image(
                image, cam, size=self.cfg.resize_image, fn="max"
            )
        
        # Rectificar pitch se configurado
        if self.cfg.rectify_pitch:
            image, cam, roll, pitch = rectify_image(
                image, cam, roll, pitch, 
                gravity_align=self.cfg.gravity_align
            )
        
        # Pad para quadrado ou múltiplo
        valid = torch.ones_like(image[:1])
        
        if self.cfg.pad_to_square:
            image, valid, cam = pad_image(
                image, valid, cam, None, None, square=True
            )
        elif self.cfg.pad_to_multiple is not None:
            image, valid, cam = pad_image(
                image, valid, cam, None, None, 
                multiple=self.cfg.pad_to_multiple
            )
        
        return image, valid.squeeze(0), cam, roll, pitch
```

**Pontos-chave:**
1. ✅ Herda de `torch.utils.data.Dataset`
2. ✅ Carrega imagens, câmeras, anotações de arquivos
3. ✅ Implementa `__getitem__` retornando dicionário compatível
4. ✅ Suporta augmentations configuráveis
5. ✅ Processa imagens (resize, pad, rectify)

---

### 4.2 DataModule - `mapper/data/indoor/data_module.py`

**Localização:** `mapper/data/indoor/data_module.py`  
**Tipo:** Novo arquivo  
**Linhas:** ~80-100  
**Descrição:** PyTorch Lightning DataModule que organiza datasets de treino/validação/teste

**Código completo:**

```python
from pathlib import Path
from typing import Optional

from ..base import DataBase
from ..schema import IndoorDataConfiguration
from .dataset import IndoorBEVDataset


class IndoorDataModule(DataBase):
    """
    PyTorch Lightning DataModule para dataset Indoor BEV.
    
    Gerencia criação e loading dos datasets de train/val/test.
    """
    
    def __init__(self, cfg: IndoorDataConfiguration):
        self.cfg = cfg
        self.datasets = {}
    
    def prepare_data(self) -> None:
        """
        Preparação de dados (download, processamento).
        Executado apenas uma vez no processo principal.
        """
        # Validar que diretórios existem
        data_dir = Path(self.cfg.data_dir)
        
        required_dirs = [
            data_dir / "images",
            data_dir / "annotations",
            data_dir / "camera_params",
            data_dir / "splits",
        ]
        
        for dir_path in required_dirs:
            if not dir_path.exists():
                raise FileNotFoundError(
                    f"Required directory not found: {dir_path}\n"
                    f"Please organize your dataset according to the expected structure."
                )
        
        # Validar que splits existem
        for split in ['train', 'val', 'test']:
            split_file = data_dir / "splits" / f"{split}.txt"
            if not split_file.exists():
                print(f"Warning: Split file not found: {split_file}")
    
    def setup(self, stage: Optional[str] = None):
        """
        Configura datasets para cada stage (fit, validate, test).
        Executado em cada processo (quando usando DDP).
        """
        if stage == "fit" or stage is None:
            self.datasets['train'] = IndoorBEVDataset(self.cfg, split='train')
            self.datasets['val'] = IndoorBEVDataset(self.cfg, split='val')
        
        if stage == "test" or stage is None:
            self.datasets['test'] = IndoorBEVDataset(self.cfg, split='test')
        
        # Logging
        for split, dataset in self.datasets.items():
            print(f"[IndoorDataModule] {split}: {len(dataset)} samples")
    
    def dataset(self, stage: str):
        """
        Retorna dataset para um stage específico.
        
        Args:
            stage: 'train', 'val', ou 'test'
        
        Returns:
            IndoorBEVDataset instance
        """
        if stage not in self.datasets:
            raise ValueError(
                f"Dataset for stage '{stage}' not initialized. "
                f"Call setup() first with appropriate stage."
            )
        
        return self.datasets[stage]
```

**Pontos-chave:**
1. ✅ Herda de `DataBase` (interface base do projeto)
2. ✅ Implementa `prepare_data()` para validação
3. ✅ Implementa `setup()` para criar datasets
4. ✅ Implementa `dataset()` para retornar dataset por stage
5. ✅ Validação de diretórios e arquivos necessários

---

### 4.3 Schema de Configuração - `mapper/data/schema.py`

**Localização:** `mapper/data/schema.py`  
**Tipo:** Modificar arquivo existente  
**Ação:** Adicionar nova dataclass ao final do arquivo

**Código a adicionar:**

```python
@dataclass(kw_only=True)
class IndoorDataConfiguration(DataConfiguration):
    """
    Configuração para dataset Indoor BEV.
    
    Args:
        data_dir: Diretório raiz do dataset
        annotation_dir: Diretório com anotações BEV (default: data_dir/annotations)
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
```

**Localização exata no arquivo:**
- Adicionar após a definição de `NuScenesDataConfiguration` (última dataclass no arquivo)
- Antes do final do arquivo

---

### 4.4 Registro em Module - `mapper/data/module.py`

**Localização:** `mapper/data/module.py`  
**Tipo:** Modificar arquivo existente  
**Ação:** Adicionar case no `get_dataset()`

**Modificação:**

```python
def get_dataset(name):
    if name == "mapillary":
        from .mapillary.data_module import MapillaryDataModule
        return MapillaryDataModule
    elif name == "nuscenes":
        from .nuscenes.data_module import NuScenesData
        return NuScenesData
    elif name == "kitti":
        from .kitti.data_module import BEVKitti360Data
        return BEVKitti360Data
    elif name == "indoor":  # ← ADICIONAR ESTAS 3 LINHAS
        from .indoor.data_module import IndoorDataModule
        return IndoorDataModule
    else:
        raise NotImplementedError(f"Dataset {name} not implemented.")
```

**Localização exata:**
- Função `get_dataset()` (linhas ~8-18)
- Adicionar o elif antes do else

---

### 4.5 Registro no ConfigStore - `mapper/mapper.py`

**Localização:** `mapper/mapper.py`  
**Tipo:** Modificar arquivo existente  
**Ação:** Adicionar imports e registros no ConfigStore

**Modificação 1: Imports (linha ~19)**

```python
from .data.schema import MIADataConfiguration, KITTIDataConfiguration, NuScenesDataConfiguration, IndoorDataConfiguration  # ← Adicionar IndoorDataConfiguration
```

**Modificação 2: ConfigStore registration (após linha ~44)**

```python
cs.store(name="mapper_indoor", node=Configuration)  # ← Config principal indoor
cs.store(group="schema/data", name="indoor", 
         node=IndoorDataConfiguration, package="data")  # ← Schema indoor
```

**Localização exata:**
- Imports: linha 19 (junto com outros imports de schema)
- ConfigStore: após linha 44 (após outros cs.store de data schemas)

---

### 4.6 Arquivo `__init__.py` - `mapper/data/indoor/__init__.py`

**Localização:** `mapper/data/indoor/__init__.py`  
**Tipo:** Novo arquivo  
**Conteúdo:** Vazio (arquivo marcador de pacote Python)

```python
# Indoor BEV Dataset Module
```

---

### 4.7 Configuração do Dataset - `mapper/conf/data/indoor.yaml`

**Localização:** `mapper/conf/data/indoor.yaml`  
**Tipo:** Novo arquivo  
**Descrição:** Configuração específica do dataset indoor

**Código completo:**

```yaml
# ============================================================================
# INDOOR BEV DATASET CONFIGURATION
# ============================================================================

name: indoor

# --- PATHS ---
data_dir: /path/to/your/indoor/dataset  # ← MODIFICAR: Caminho do seu dataset
annotation_dir: null  # Default: ${data_dir}/annotations

# --- CLASSES ---
num_classes: 8  # Dataset warehouse/industrial

# Lista de nomes das classes
class_names:
  - obstrucao       # Classe 0: Chapas, lixeiras, cones
  - empilhadeira    # Classe 1: Outras empilhadeiras/transpaleteiras
  - carga           # Classe 2: Pallets, caixas, carrinhos
  - maquina         # Classe 3: Máquinas/esteiras industriais
  - humano          # Classe 4: Pessoas no ambiente
  - navegavel       # Classe 5: Área livre para navegação
  - estrutura       # Classe 6: Colunas, paredes, pilares
  - portapalete     # Classe 7: Racks de armazenamento

# Mapeamento de classes (opcional)
# Exemplo: [0, 0, 1, 2] mapeia classes 0,1→0, 2→1, 3→2
class_mapping: null

# --- BEV GRID PARAMETERS ---
pixel_per_meter: 6           # Resolução: 6 pixels/metro = ~16.7 cm/pixel
crop_size_meters:5           # Resolução: 5 pixels/metro = 20 cm/pixel
crop_size_meters: 15         # Área BEV: 15m profundidade
# --- CAMERA PARAMETERS ---
target_focal_length: null    # Normalização de focal (null = desabilitado)
resize_image: 512            # Resize de imagem para 512px (lado maior)
pad_to_square: true          # Pad para quadrado
pad_to_multiple: null        # Pad para múltiplo (null = desabilitado)
rectify_pitch: true          # Corrigir inclinação da câmera
gravity_align: true          # Alinhar com gravidade

# --- FILTERS (Optional) ---
room_types: null             # Filtrar por tipo de ambiente: ['office', 'corridor']
floor_levels: null           # Filtrar por andar: [0, 1, 2]
percentage: 1.0              # Fração do dataset a usar (1.0 = 100%)

# --- AUGMENTATIONS ---
augmentations:
  enabled: true              # Habilitar augmentations no treino
  
  # Color jitter
  brightness: 0. (adaptado para iluminação industrial - LED/fluorescente)
  brightness: 0.2            # Variação de brilho controlada
  contrast: 0.2              # Variação de contraste moderada
  saturation: 0.15           # Saturação baixa (ambiente industrial)
  hue: 0.05                  # Matiz mínima (cores consistentes)
  # Flip horizontal
  random_flip: 0.5           # 50% chance de flip (simetria de ambientes)
  
  # Crop (geralmente desabilitado para BEV)
  random_resized_crop: false
  
  # Gaussian noise (simula ruído de câmeras industriais)
  gaussian_noise:
    enabled: true
    mean: 0.0
    std: 0.03                # Ruído leve (câmeras industriais são geralmente boas)
  
  # Brightness/Contrast adicional
  brightness_contrast:
    enabled: true
    brightness_factor: 0.15
    contrast_factor: 0.15

# --- DATA LOADING ---
loading:
  train:
    batch_size: 16           # ← MODIFICAR: Ajustar conforme GPU
    num_workers: 8           # Threads de loading
  
  val:
    batch_size: 16
    num_workers: 4
  
  test:
    batch_size: 8
    num_workers: 2

# --- MISC ---
seed: 42                     # Seed para reprodutibilidade
random: true                 # Randomização de exemplos no treino
num_threads: null            # Threads PyTorch (null = auto)
```

**Valores a modificar:**
1. `data_dir`: Caminho do seu dataset
2. `num_classes`: Número de classes
3. `class_names`: Nomes das suas classes
4. `batch_size`: Conforme capacidade da GPU
5. `pixel_per_meter`, `crop_size_meters`: Conforme ambiente

---

### 4.8 Configuração Principal - `mapper/conf/mapper_indoor.yaml`

**Localização:** `mapper/conf/mapper_indoor.yaml`  
**Tipo:** Novo arquivo  
**Descrição:** Configuração principal do experimento indoor

**Código completo:**

```yaml
# ============================================================================
# MAPPER INDOOR - MAIN CONFIGURATION
# ============================================================================

defaults:
  - schema/data: indoor       # Schema de validação
  - data: indoor              # Configuração do dataset
  - model: mapper             # Modelo base (MapPerceptionNet)
  - training                  # Configuração de treinamento
  - _self_                    # Override com configs abaixo

# --- EXPERIMENT ---
experiment:
  name: MIA_DINOv2_Mapper_Warehouse  # Nome do experimento (WandB, checkpoints)

# --- MODEL OVERRIDES ---
model:
  # Número de classes (deve coincidir com data.num_classes)
  num_classes: ${data.num_classes}
  
  # BEV grid parameters (adaptado para warehouse/industrial)
  z_max: 15                   # Profundidade máxima: 15 metros
  x_max: 6                    # Largura lateral: 6m cada lado (total 12m - cobertura de corredor)
  z_min: 0.5                  # Profundidade mínima: 0.5m
  pixel_per_meter: ${data.pixel_per_meter}  # Herda do dataset
  
  # Scale bins para projeção polar
  num_scale_bins: 24          # Menos bins que outdoor (menor range de profundidade)
  scale_range: [0, 7]         # Range de escala log ajustado
  
  # Latent dimension (features do encoder)
  latent_dim: 128             # Mantém padrão
  
  # Loss function
  loss:
    num_classes: ${..num_classes}
    
    # Loss weights
    xent_weight: 1.0          # Cross-entropy weight
    dice_weight: 1.0          # Dice loss weight
    
    # Focal loss (útil para classes desbalanceadas)
    focal_loss: true          # ← Ativar se classes muito desbalanceadas
    focal_loss_gamma: 2.0     # Gamma para focal loss
    
    # Máscaras de validação
    requires_frustrum: true   # Usar máscara de frustrum (campo de visão)
    requires_flood_mask: true # Usar flood mask (regiões válidas)
    
    # Class weights (ajustar conforme distribuição real do dataset)
    # Valores iniciais baseados em frequência esperada em warehouse
    class_weights:
      - 1.5   # obstrucao (moderado - objetos temporários)
      - 2.0   # empilhadeira (raro - poucas empilhadeiras simultaneamente)
      - 1.2   # carga (comum - pallets e caixas)
      - 2.5   # maquina (raro - máquinas fixas)
      - 3.0   # humano (muito raro - segurança, poucos operadores)
      - 0.6   # navegavel (muito frequente - maior parte do chão)
      - 1.0   # estrutura (comum - colunas e paredes)
      - 1.3   # portapalete (comum - racks de armazenamento)
    # ⚠️ IMPORTANTE: Calcular weights reais usando script da seção 8.3
    
    # Label smoothing (regularização)
    label_smoothing: 0.1

# --- TRAINING OVERRIDES ---
training:
  num_classes: ${model.num_classes}
  
  # Checkpoint pré-treinado (fine-tuning)
  checkpoint: null            # ← MODIFICAR: Path do checkpoint pré-treinado
  # Exemplo: /path/to/mia_pretrained.ckpt
  
  finetune: false             # ← Mudar para true se usar checkpoint
  
  # Learning rate
  lr: 0.001                   # LR inicial (0.0001 se fine-tuning)
  
  # Learning rate scheduler
  lr_scheduler:
    name: "CosineAnnealingLR"
    args:
      T_max: $total_epochs    # Total de epochs (substituído automaticamente)
      eta_min: 0.0000001      # LR mínimo
  
  # Evaluation mode
  eval: false                 # true = apenas avaliação
  save_dir: eval_results      # Diretório de resultados de avaliação
  
  # Trainer parameters
  trainer:
    max_epochs: 100           # ← MODIFICAR: Número de epochs
    precision: bf16-mixed     # Precisão mista (economiza memória)
    accelerator: gpu
    devices: 1                # ← MODIFICAR: Número de GPUs
    strategy: ddp_find_unused_parameters_true  # Strategy para DDP
    
    # Validation
    val_check_interval: 1.0   # Validar a cada epoch
    check_val_every_n_epoch: 1
    
    # Logging
    log_every_n_steps: 50
    
    # Limits (para debug)
    # limit_train_batches: 100  # Descomentar para testar rápido
    # limit_val_batches: 10
  
  # Checkpointing
  checkpointing:
    dirpath: checkpoints/indoor/  # Diretório de checkpoints
    monitor: val/total/loss       # Métrica para monitorar
    save_top_k: 3                 # Salvar 3 melhores checkpoints
    mode: min                     # Minimizar loss
    save_last: true               # Salvar último checkpoint
    filename: "{epoch}-{val/total/loss:.4f}"  # Nome do arquivo
```

**Valores a modificar:**
1. `training.checkpoint`: Path do modelo pré-treinado (se usar fine-tuning)
2. `training.finetune`: `true` se usar checkpoint
3. `training.lr`: `0.0001` se fine-tuning, `0.001` do zero
4. `training.trainer.max_epochs`: Conforme tempo disponível
5. `training.trainer.devices`: Número de GPUs
6. `model.loss.class_weights`: Ajustar aos seus dados

---

## 5. ESTRUTURA DE DADOS NECESSÁRIA

### 5.1 Organização de Diretórios

```
/path/to/your/indoor/dataset/
│
├── images/                      # Imagens RGB
│   ├── room1_frame001.jpg
│   ├── room1_frame002.jpg
│   ├── room2_frame001.jpg
│   └── ...
│
├── annotations/                 # Máscaras BEV (ground truth)
│   ├── room1_frame001.npy      # Shape: (H, W, num_classes)
│   ├── room1_frame002.npy
│   ├── room2_frame001.npy
│   └── ...
│
├── camera_params/               # Parâmetros intrínsecos da câmera
│   ├── room1_frame001.json
│   ├── room1_frame002.json
│   ├── room2_frame001.json
│   └── ...
│
├── flood_masks/                 # (OPCIONAL) Máscaras de visibilidade
│   ├── room1_frame001.npy      # Shape: (H, W), valores: 0=observável, 1=não
│   ├── room1_frame002.npy
│   └── ...
│
└── splits/                      # Divisão train/val/test
    ├── train.txt               # Lista de frame_names (1 por linha)
    ├── val.txt
    └── test.txt
```

### 5.2 Formato das Anotações BEV

**Arquivo:** `annotations/frame_name.npy`

**Formato:**
```python
# Shape: (H, W, num_classes)
# H: Altura do grid BEV em pixels = z_max × pixel_per_meter
# W: Largura do grid BEV em pixels = 2 × x_max × pixel_per_meter
# num_classes: Número de classes semânticas

# Exemplo: 15m depth, 16m width, 6 px/m, 9 classes
# Shape: (90, 96, 9)

# One-hot encoding:
# seg_mask[y, x, :] = [0, 0, 1, 0, 0, 0, 0, 0, 0]  # Classe 2 no pixel (y,x)

# ❗ IMPORTANTE: Cada pixel deve ter exatamente UMA classe ativa
# Sum ao longo do eixo de classes deve ser 1:
# assert seg_mask.sum(axis=2).all() == 1.0
```

**Script para criar anotação exemplo:**

```python
import numpy as np

def create_bev_annotation_example():
    """Cria exemplo de anotação BEV para warehouse."""
    
    # Parâmetros
    z_max = 15  # metros
    x_max = 6   # metros (total 12m de largura)
    pixel_per_meter = 5
    num_classes = 8  # Classes warehouse
    
    # Dimensões do grid
    H = int(z_max * pixel_per_meter)  # 75 pixels
    W = int(2 * x_max * pixel_per_meter)  # 60 pixels
    
    # Inicializar com classe 5 (navegavel) em todos os lugares
    seg_mask = np.zeros((H, W, num_classes), dtype=np.uint8)
    seg_mask[:, :, 5] = 1  # Toda área navegável inicialmente
    
    # Adicionar estruturas laterais (classe 6: estrutura) - paredes/colunas
    seg_mask[:, 0:3, 5] = 0     # Remover navegavel
    seg_mask[:, 0:3, 6] = 1     # Adicionar estrutura esquerda
    seg_mask[:, -3:, 5] = 0
    seg_mask[:, -3:, 6] = 1     # Adicionar estrutura direita
    
    # Adicionar porta-palete (classe 7) nas laterais
    seg_mask[10:30, 5:10, 5] = 0
    seg_mask[10:30, 5:10, 7] = 1  # Rack esquerdo
    
    # Adicionar carga (classe 2: pallet) no chão
    center_y, center_x = H // 2, W // 2
    seg_mask[center_y-3:center_y+3, center_x-3:center_x+3, 5] = 0
    seg_mask[center_y-3:center_y+3, center_x-3:center_x+3, 2] = 1
    
    # Validar: cada pixel tem exatamente uma classe
    assert (seg_mask.sum(axis=2) == 1).all(), "Each pixel must have exactly one class"
    
    # Salvar
    np.save('example_annotation.npy', seg_mask)
    print(f"Created annotation with shape {seg_mask.shape}")
    
    return seg_mask
```

### 5.3 Formato dos Parâmetros de Câmera

**Arquivo:** `camera_params/frame_name.json`

**Formato:**
```json
{
  "width": 1920,
  "height": 1080,
  "fx": 800.0,
  "fy": 800.0,
  "cx": 960.0,
  "cy": 540.0,
  "roll": 0.0,
  "pitch": 0.0,
  "yaw": 0.0
}
```

**Descrição dos campos:**
- `width`, `height`: Resolução da imagem em pixels
- `fx`, `fy`: Distância focal em pixels (eixos x e y)
- `cx`, `cy`: Centro óptico em pixels (geralmente width/2, height/2)
- `roll`, `pitch`, `yaw`: Orientação da câmera em radianos

**Como obter parâmetros:**
- Calibração de câmera (OpenCV, ROS)
- EXIF de imagens (se disponível)
- Valores padrão: fx=fy=focal_pixels, cx=width/2, cy=height/2

### 5.4 Formato dos Splits

**Arquivo:** `splits/train.txt`, `val.txt`, `test.txt`

**Formato:**
```
room1_frame001
room1_frame002
room1_frame003
room2_frame001
room2_frame010
...
```

- Um `frame_name` por linha (sem extensão)
- `frame_name` deve corresponder aos arquivos em `images/`, `annotations/`, `camera_params/`

**Script para criar splits:**

```python
import random
from pathlib import Path

def create_splits(data_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Cria splits train/val/test."""
    
    data_dir = Path(data_dir)
    image_dir = data_dir / "images"
    
    # Listar todos os frames
    frame_names = [f.stem for f in image_dir.glob("*.jpg")]
    
    # Shuffle
    random.seed(42)
    random.shuffle(frame_names)
    
    # Split
    n = len(frame_names)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_frames = frame_names[:n_train]
    val_frames = frame_names[n_train:n_train + n_val]
    test_frames = frame_names[n_train + n_val:]
    
    # Salvar
    splits_dir = data_dir / "splits"
    splits_dir.mkdir(exist_ok=True)
    
    for split_name, frames in [('train', train_frames), ('val', val_frames), ('test', test_frames)]:
        with open(splits_dir / f"{split_name}.txt", 'w') as f:
            f.write('\n'.join(frames))
    
    print(f"Created splits: train={len(train_frames)}, val={len(val_frames)}, test={len(test_frames)}")
```

---

## 6. PROCESSO DE IMPLEMENTAÇÃO

### 6.1 Checklist Passo-a-Passo

#### Fase 1: Preparação dos Dados
- [ ] **1.1** Organizar imagens no diretório `images/`
- [ ] **1.2** Gerar/converter anotações BEV para formato `.npy` (H, W, C) one-hot
- [ ] **1.3** Calibrar câmeras ou usar parâmetros padrão
- [ ] **1.4** Criar arquivos JSON com parâmetros de câmera
- [ ] **1.5** Criar splits train/val/test (70/15/15 sugerido)
- [ ] **1.6** (Opcional) Gerar flood masks se necessário
- [ ] **1.7** Validar estrutura de diretórios

#### Fase 2: Implementação do Código
- [ ] **2.1** Criar diretório `mapper/data/indoor/`
- [ ] **2.2** Criar `mapper/data/indoor/__init__.py`
- [ ] **2.3** Criar `mapper/data/indoor/dataset.py` (usar template 4.1)
- [ ] **2.4** Criar `mapper/data/indoor/data_module.py` (usar template 4.2)
- [ ] **2.5** Modificar `mapper/data/schema.py` (adicionar IndoorDataConfiguration)
- [ ] **2.6** Modificar `mapper/data/module.py` (adicionar case "indoor")
- [ ] **2.7** Modificar `mapper/mapper.py` (imports e ConfigStore)

#### Fase 3: Configuração
- [ ] **3.1** Criar `mapper/conf/data/indoor.yaml` (usar template 4.7)
- [ ] **3.2** Ajustar parâmetros em `indoor.yaml`:
  - [ ] `data_dir`
  - [ ] `num_classes`
  - [ ] `class_names`
  - [ ] `pixel_per_meter`
  - [ ] `crop_size_meters`
  - [ ] `batch_size`
- [ ] **3.3** Criar `mapper/conf/mapper_indoor.yaml` (usar template 4.8)
- [ ] **3.4** Ajustar parâmetros em `mapper_indoor.yaml`:
  - [ ] `z_max`, `x_max`
  - [ ] `class_weights`
  - [ ] `training.checkpoint` (se fine-tuning)
  - [ ] `training.finetune`
  - [ ] `training.lr`

#### Fase 4: Validação
- [ ] **4.1** Testar import do módulo:
  ```bash
  python -c "from mapper.data.indoor.dataset import IndoorBEVDataset; print('OK')"
  ```
- [ ] **4.2** Testar loading de configuração:
  ```bash
  python -m mapper.mapper --config-name=mapper_indoor --help
  ```
- [ ] **4.3** Testar loading de um batch:
  ```bash
  python -m mapper.mapper --config-name=mapper_indoor training.trainer.max_epochs=1 training.trainer.limit_train_batches=1
  ```
- [ ] **4.4** Validar shapes dos tensors:
  - [ ] `image`: (C, H, W) onde C=3
  - [ ] `seg_masks`: (H_bev, W_bev, num_classes)
  - [ ] `camera`: Camera object com 6 parâmetros
  - [ ] `flood_masks`: (H_bev, W_bev)

#### Fase 5: Treinamento
- [ ] **5.1** (Opcional) Baixar checkpoint pré-treinado (MIA/KITTI/nuScenes)
- [ ] **5.2** Configurar checkpoint e finetune em `mapper_indoor.yaml`
- [ ] **5.3** Executar treinamento inicial (poucas epochs):
  ```bash
  python -m mapper.mapper --config-name=mapper_indoor training.trainer.max_epochs=5
  ```
- [ ] **5.4** Monitorar métricas (IoU, loss, mAP)
- [ ] **5.5** Ajustar hiperparâmetros se necessário:
  - [ ] Learning rate
  - [ ] Class weights
  - [ ] Augmentations
  - [ ] Batch size
- [ ] **5.6** Treinamento completo

#### Fase 6: Avaliação
- [ ] **6.1** Avaliar no conjunto de teste:
  ```bash
  python -m mapper.mapper --config-name=mapper_indoor training.eval=true training.checkpoint=/path/to/best.ckpt
  ```
- [ ] **6.2** Analisar resultados por classe
- [ ] **6.3** Visualizar predições BEV
- [ ] **6.4** Identificar classes problemáticas
- [ ] **6.5** Iterar: ajustar dados, augmentations, ou arquitetura

### 6.2 Comandos Úteis

#### Teste de Configuração
```bash
# Validar YAML sem executar
python -m mapper.mapper --config-name=mapper_indoor --cfg job

# Mostrar configuração resolvida
python -m mapper.mapper --config-name=mapper_indoor --cfg job | less
```

#### Treinamento do Zero
```bash
# Treinamento completo
python -m mapper.mapper --config-name=mapper_indoor \
    training.trainer.max_epochs=100

# Com logging WandB desabilitado (para teste local)
python -m mapper.mapper --config-name=mapper_indoor \
    training.trainer.max_epochs=10 \
    training.trainer.logger=false
```

#### Fine-tuning
```bash
# Fine-tuning a partir de checkpoint pré-treinado
python -m mapper.mapper --config-name=mapper_indoor \
    training.checkpoint=/path/to/pretrained.ckpt \
    training.finetune=true \
    training.lr=0.0001 \
    training.trainer.max_epochs=50
```

#### Avaliação
```bash
# Avaliar checkpoint no test set
python -m mapper.mapper --config-name=mapper_indoor \
    training.eval=true \
    training.checkpoint=/path/to/best.ckpt \
    training.save_dir=results/indoor_test
```

#### Debug
```bash
# Executar apenas 10 batches de treino e 5 de validação
python -m mapper.mapper --config-name=mapper_indoor \
    training.trainer.limit_train_batches=10 \
    training.trainer.limit_val_batches=5 \
    training.trainer.max_epochs=1
```

### 6.3 Resolução de Problemas Comuns

#### Erro: "Dataset indoor not implemented"
**Causa:** `get_dataset()` não inclui case "indoor"  
**Solução:** Verificar modificação em `mapper/data/module.py` (seção 4.4)

#### Erro: "No module named 'mapper.data.indoor'"
**Causa:** Módulo indoor não criado ou `__init__.py` ausente  
**Solução:** Criar diretório `mapper/data/indoor/` com `__init__.py`

#### Erro: "ConfigStore: missing IndoorDataConfiguration"
**Causa:** Schema não registrado no ConfigStore  
**Solução:** Verificar modificações em `mapper/mapper.py` (seção 4.5)

#### Erro: Shape mismatch em seg_masks
**Causa:** Anotações não têm o shape correto (H, W, C)  
**Solução:** 
- Validar `seg_masks.shape[-1] == num_classes`
- Verificar que H, W correspondem ao grid BEV esperado

#### Erro: "Camera parameter not found"
**Causa:** Arquivo JSON de câmera faltando ou nome incorreto  
**Solução:**
- Verificar nomenclatura: `camera_params/{frame_name}.json`
- Validar que frame_name corresponde ao split

#### Warning: "Split file not found"
**Causa:** Arquivo `splits/{split}.txt` não existe  
**Solução:** Criar splits com script da seção 5.4

#### Loss = NaN
**Causas possíveis:**
1. Learning rate muito alto → Reduzir para 1e-4 ou 1e-5
2. Class weights mal configurados → Normalizar ou usar None
3. Dados com valores infinitos → Validar anotações

**Solução:** 
```yaml
training:
  lr: 0.0001
model:
  loss:
    class_weights: null  # Desabilitar temporariamente
```

#### IoU muito baixo (<0.1)
**Causas possíveis:**
1. Classes desbalanceadas → Usar class_weights e focal loss
2. Modelo não convergiu → Treinar mais epochs
3. Dados de treino insuficientes → Coletar mais dados
4. Augmentations muito agressivas → Reduzir intensidade

**Solução:**
```yaml
model:
  loss:
    focal_loss: true
    class_weights: [ajustar conforme distribuição]
training:
  trainer:
    max_epochs: 150  # Aumentar
```

---

## 7. FINE-TUNING vs TREINAMENTO DO ZERO

### 7.1 Quando Usar Fine-tuning

**Vantagens:**
- ✅ Converge mais rápido (5-10× menos epochs)
- ✅ Requer menos dados (pode funcionar com ~1000 imagens)
- ✅ Features visuais já aprendidas (DINOv2 pré-treinado)
- ✅ Melhor generalização com datasets pequenos

**Desvantagens:**
- ⚠️ Requer checkpoint compatível (mesmo backbone)
- ⚠️ Bias para domínio outdoor (se usar checkpoint outdoor)

**Recomendado quando:**
- Dataset indoor tem < 5000 imagens
- Ambiente indoor tem elementos visuais semelhantes a outdoor (texturas, geometria)
- Tempo de treinamento limitado
- GPU limitada

### 7.2 Quando Treinar do Zero

**Vantagens:**
- ✅ Sem bias de domínio outdoor
- ✅ Adaptação perfeita ao domínio indoor
- ✅ Controle total sobre aprendizado

**Desvantagens:**
- ⚠️ Requer muito mais dados (>10000 imagens)
- ⚠️ Convergência lenta (100-300 epochs)
- ⚠️ Risco de overfitting em datasets pequenos

**Recomendado quando:**
- Dataset indoor tem > 10000 imagens bem anotadas
- Ambiente indoor muito diferente de outdoor
- Recursos computacionais abundantes

### 7.3 Configuração para Fine-tuning

**Checkpoint pré-treinado:**
- Download do HuggingFace: [mapitanywhere/mapper](https://huggingface.co/mapitanywhere/mapper)
- Ou usar checkpoint KITTI/nuScenes treinado localmente

**Modificações em `mapper_indoor.yaml`:**

```yaml
training:
  # Path do checkpoint pré-treinado
  checkpoint: /path/to/pretrained/mapper_mia.ckpt
  
  # Habilitar fine-tuning
  finetune: true
  
  # Learning rate reduzido (10x menor que treino do zero)
  lr: 0.0001
  
  # Menos epochs necessários
  trainer:
    max_epochs: 50  # vs 100-150 do zero
```

**Estratégias de Fine-tuning:**

1. **Freeze Encoder (primeira fase):**
   ```python
   # Modificar mapper/module.py após linha 30
   if cfg.training.freeze_encoder:
       for param in self.model.image_encoder.parameters():
           param.requires_grad = False
   ```
   
   ```yaml
   # mapper_indoor.yaml
   training:
     freeze_encoder: true
     trainer:
       max_epochs: 30
   ```

2. **Unfreeze Encoder (segunda fase):**
   ```yaml
   training:
     freeze_encoder: false
     checkpoint: /path/to/checkpoint_from_phase1.ckpt
     lr: 0.00001  # 10x menor ainda
     trainer:
       max_epochs: 20
   ```

### 7.4 Configuração para Treino do Zero

```yaml
training:
  checkpoint: null     # Sem checkpoint
  finetune: false      # Desabilitado
  lr: 0.001            # LR maior
  
  trainer:
    max_epochs: 150    # Mais epochs
    
model:
  image_encoder:
    backbone:
      pretrained: true  # Ainda usar DINOv2 pré-treinado (ImageNet)
      frozen: false     # Mas permitir fine-tune
```

---

## 8. OTIMIZAÇÕES E BOAS PRÁTICAS

### 8.1 Performance de Treinamento

#### GPU Memory
```yaml
# Reduzir uso de memória
data:
  loading:
    train:
      batch_size: 8  # Reduzir batch size

training:
  trainer:
    precision: bf16-mixed  # Usar mixed precision
    gradient_clip_val: 1.0  # Clip gradients
```

#### Velocidade de Loading
```yaml
data:
  loading:
    train:
      num_workers: 8  # Aumentar threads (ajustar conforme CPU)
      prefetch_factor: 2  # Pre-fetch batches
```

### 8.2 Regularização

```yaml
# Combate overfitting
model:
  loss:
    label_smoothing: 0.1  # Label smoothing
    
  segmentation_head:
    dropout_rate: 0.3  # Aumentar dropout

data:
  augmentations:
    enabled: true
    # Augmentations mais agressivas
```

### 8.3 Class Balancing

# Nomes das classes warehouse
```python 
CLASS_NAMES = [
    "obstrucao", "empilhadeira", "carga", "maquina",
    "humano", "navegavel", "estrutura", "portapalete"
]

def calculate_class_weights(data_dir, num_classes=8):
    """Calcula pesos de classes baseado em frequência do dataset warehouse."""
    annotation_dir = Path(data_dir) / "annotations"
    
    # Contar pixels por classe
    class_counts = np.zeros(num_classes)
    
    print("Processando anotações...")
    for anno_file in annotation_dir.glob("*.npy"):
        seg_mask = np.load(anno_file)  # (H, W, C)
        class_counts += seg_mask.sum(axis=(0, 1))
    
    # Calcular pesos inversamente proporcionais
    total_pixels = class_counts.sum()
    class_frequencies = class_counts / total_pixels
    class_weights = 1.0 / (class_frequencies + 1e-6)
    
    # Normalizar para média = 1
    class_weights = class_weights / class_weights.mean()
    
    # Exibir resultados
    print("\n" + "="*60)
    print("DISTRIBUIÇÃO DE CLASSES NO DATASET WAREHOUSE")
    print("="*60)
    for i, (name, freq, weight) in enumerate(zip(CLASS_NAMES, class_frequencies, class_weights)):
        print(f"Classe {i} ({name:15s}): {freq*100:5.2f}% | Weight: {weight:.3f}")
    print("="*60)
    
    print("\nClass weights para YAML:")
    print("class_weights:")
    for i, (name, weight) in enumerate(zip(CLASS_NAMES, class_weights)):
        print(f"  - {weight:.3f}   # {name}")
    
    return class_weights.tolist()

# Executar:
# weights = calculate_class_weights('/path/to/dataset', num_classes=8)frequencies)
    print("Class weights:", class_weights.tolist())
    
    return class_weights.tolist()

# Usar em mapper_indoor.yaml:
# class_weights: [resultado do script]
```

### 8.4 Validação de Dados

**Script de validação pré-treino:**

```python
from pathlib import Path
import numpy as np
import json

def validate_dataset(data_dir):
    """Valida integridade do dataset."""
    data_dir = Path(data_dir)
    
    errors = []
    
    # Carregar splits
    for split in ['train', 'val', 'test']:
        split_file = data_dir / "splits" / f"{split}.txt"
        if not split_file.exists():
            errors.append(f"Missing split file: {split}.txt")
            continue
        
        with open(split_file) as f:
            frame_names = [l.strip() for l in f]
        
        for frame_name in frame_names:
            # Validar imagem
            img_path = data_dir / "images" / f"{frame_name}.jpg"
            if not img_path.exists():
                errors.append(f"Missing image: {frame_name}.jpg")
            
            # Validar anotação
            anno_path = data_dir / "annotations" / f"{frame_name}.npy"
            if not anno_path.exists():
                errors.append(f"Missing annotation: {frame_name}.npy")
            else:
                seg_mask = np.load(anno_path)
                if seg_mask.ndim != 3:
                    errors.append(f"Invalid annotation shape: {frame_name} has {seg_mask.ndim}D")
                
                # Validar one-hot
                if not np.allclose(seg_mask.sum(axis=2), 1.0):
                    errors.append(f"Annotation not one-hot: {frame_name}")
            
            # Validar câmera
            cam_path = data_dir / "camera_params" / f"{frame_name}.json"
            if not cam_path.exists():
                errors.append(f"Missing camera params: {frame_name}.json")
            else:
                with open(cam_path) as f:
                    cam = json.load(f)
                required_keys = ['width', 'height', 'fx', 'fy', 'cx', 'cy']
                for key in required_keys:
                    if key not in cam:
                        errors.append(f"Missing camera param '{key}': {frame_name}")
    
    if errors:
        print(f"Found {len(errors)} errors:")
        for err in errors[:20]:  # Mostrar primeiros 20
            print(f"  - {err}")
    else:
        print("✅ Dataset validation passed!")
    
    return len(errors) == 0
```

---

## 9. MÉTRICAS E MONITORAMENTO

### 9.1 Métricas Calculadas

O modelo automaticamente calcula e loga:

1. **Pixel Accuracy**: Acurácia pixel-a-pixel
2. **mIoU Observable**: IoU médio em regiões observáveis pela câmera
3. **mIoU Non-Observable**: IoU médio em regiões não-observáveis (predição além do FOV)
4. **mAP**: Mean Average Precision multi-label
5. **IoU per-class**: IoU individual para cada classe
6. **Loss components**: Cross-entropy, Dice, Total

### 9.2 Interpretação de Métricas

**Pixel Accuracy:**
- Boa: > 0.85
- Aceitável: 0.70 - 0.85
- Ruim: < 0.70
- ⚠️ **Cuidado:** Pode ser enganosa com classes desbalanceadas

**mIoU Observable:**
- Excelente: > 0.70
- Bom: 0.50 - 0.70
- Aceitável: 0.30 - 0.50
- Ruim: < 0.30

**mIoU Non-Observable:**
- Geralmente menor que observable (~0.3-0.5)
- Indica capacidade de predizer além do campo de visão

**IoU per-class:**
- Identificar classes problemáticas (IoU < 0.3)
- Ajustar class_weights ou coletar mais dados dessas classes

### 9.3 WandB Logging

Se habilitado (padrão), métricas são enviadas para Weights & Biases:

```yaml
# Desabilitar WandB (para desenvolvimento local)
training:
  trainer:
    logger: false

# Ou configurar WandB
# Modificar mapper/mapper.py linha ~80:
logger = WandbLogger(
    name=exp_name_with_time,
    entity="seu-usuario",  # ← Modificar
    project="indoor-bev",  # ← Modificar
)
```

---
 - Ambiente Industrial
1. 🚀 Exportar modelo para inferência (ONNX/TorchScript)
2. 🚀 Criar pipeline de pós-processamento (filtros de navegabilidade)
3. 🚀 Integrar com sistema de controle da empilhadeira/AGV
4. 🚀 Monitorar performance em warehouse real
5. 🚀 Coletar dados de casos extremos:
   - Condições de iluminação variável
   - Novos tipos de carga/obstáculos
   - Interferência de outros robôs
6. 🚀 Retreinar periodicamente com dados de produção.4
3. ✅ Testar loading de um batch
4. ✅ Executar 1 epoch de treino e validação
5. ✅ Verificar que métricas são calculadas

### 10.2 Experimentos Iniciais
1. 🔬 Baseline sem fine-tuning (10-20 epochs)
2. 🔬 Fine-tuning com checkpoint MIA (30-50 epochs)
3. 🔬 Ajuste de class_weights baseado em frequências
4. 🔬 Teste de diferentes augmentations
5. 🔬 Grid search de learning rate

### 10.3 Otimização
1. 📈 Análise de erros por classe
2. 📈 Ajuste de z_max, x_max baseado em resultados
3. 📈 Teste de diferentes resoluções (pixel_per_meter)
4. 📈 Ensemble de checkpoints (se disponível)

### 10.4 Produção
1. 🚀 Exportar modelo para inferência
2. 🚀 Criar pipeline de pós-processamento
3. 🚀 Integrar com sistema de navegação robótica
4. 🚀 Monitorar performance em ambiente real
5. 🚀 Coletar dados edge cases e retreinar

---

## 11. REFERÊNCIAS E RECURSOS

### 11.1 Documentação

- **MIA Original:** https://github.com/MapItAnywhere/MapItAnywhere
- **DINOv2:** https://github.com/facebookresearch/dinov2
- **PyTorch Lightning:** https://lightning.ai/docs/pytorch/
- **Hydra:** https://hydra.cc/docs/intro/

### 11.2 Papérs Relevantes

1. **Map It Anywhere** (Paper original do MIA)
2. **DINOv2: Learning Robust Visual Features** (Backbone)
3. **Lift, Splat, Shoot** (BEV projection inspiration)

### 11.3 Datasets BEV Relacionados

- **KITTI-360:** http://www.cvlibs.net/datasets/kitti-360/
- **nuScenes:** https://www.nuscenes.org/
- **Mapillary:** https://www.mapillary.com/dataset/vistas

### 11.4 Contato e Suporte

- **Issues:** Github do projeto original
- **Comunidade:** Discord/Slack da equipe (se existir)

---

## 12. RESUMO EXECUTIVO

### 12.1 Arquivos a Criar (7 arquivos)

| Arquivo | Localização | Linhas | Descrição |
|---------|------------|--------|-----------|
| ✅ dataset.py | `mapper/data/indoor/` | ~250 | Dataset principal |
| ✅ data_module.py | `mapper/data/indoor/` | ~100 | DataModule Lightning |
| ✅ __init__.py | `mapper/data/indoor/` | 1 | Marcador de pacote |
| ✅ indoor.yaml | `mapper/conf/data/` | ~80 | Config dataset |
| ✅ mapper_indoor.yaml | `mapper/conf/` | ~100 | Config experimento |

### 12.2 Arquivos a Modificar (3 arquivos)de câmeras frontais e gerar mapas BEV semânticos de ambientes warehouse/industriais, identificando 8 classes críticas para navegação autônoma de empilhadeiras e AGVs: áreas navegáveis, obstáculos (obstruções, cargas, máquinas), entidades dinâmicas (humanos, outras empilhadeiras) e estruturas fixas (paredes, porta-paletes

| Arquivo | Localização | Modificação | Linhas |
|---------|------------|-------------|--------|
| ⚠️ schema.py | `mapper/data/` | Adicionar IndoorDataConfiguration | +25 |
| ⚠️ module.py | `mapper/data/` | Adicionar case "indoor" | +3 |
| ⚠️ mapper.py | `mapper/` | Imports + ConfigStore | +2 |

### 12.3 Esforço Estimado

- **Preparação de dados:** 2-5 dias (depende de dataset existente)
- **Implementação de código:** 0.5-1 dia (usando templates)
- **Configuração e testes:** 0.5 dia
- **Primeiro treinamento:** 1-2 dias (depende de GPU)
- **Iteração e otimização:** 1-2 semanas

**Total:** ~1-3 semanas do início ao modelo funcional

### 12.4 Conclusão

Este relatório documenta todas as modificações necessárias para adaptar o modelo MIA para datasets indoor BEV. Seguindo os templates e checklists fornecidos, a implementação é direta e requer principalmente:

1. ✅ Organização correta dos dados
2. ✅ Criação de 3 arquivos Python (~300 linhas usando templates)
3. ✅ Modificação mínima de arquivos existentes (~30 linhas)
4. ✅ Configuração de parâmetros em YAMLs

O modelo resultante será capaz de processar imagens indoor e gerar mapas BEV semânticos, adaptado às características específicas de ambientes fechados (menor área, maior resolução, classes室内).

**Sucesso no projeto!** 🚀

---

**Fim do Relatório**

*Versão 1.0 - 12 de fevereiro de 2026*
