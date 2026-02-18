from huggingface_hub import snapshot_download
from pathlib import Path

repo_id = "mapitanywhere/mapper"
local_dir = Path("weights")

local_dir.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
)

print("✅ Pesos baixados em:", local_dir.resolve())
