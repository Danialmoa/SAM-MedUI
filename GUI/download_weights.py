"""Auto-download model weights from HuggingFace if not present."""

import os
import sys
from pathlib import Path

HF_REPO_ID = "AidaAIDL/SAM_MEDUI"

# Files to download and where to put them (relative to project root)
WEIGHT_FILES = {
    "best_model.pth": "checkpoints/best_model.pth",
    "yolo_best.pt": "checkpoints/yolo_best.pt",
}


def get_project_root():
    """Get the project root directory (parent of GUI/)."""
    return Path(__file__).resolve().parent.parent


def check_weights(root: Path) -> list:
    """Return list of missing weight files."""
    missing = []
    for hf_name, local_path in WEIGHT_FILES.items():
        full_path = root / local_path
        if not full_path.exists():
            missing.append((hf_name, full_path))
    return missing


def download_weights(root: Path, missing: list):
    """Download missing weights from HuggingFace Hub."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("\nhuggingface_hub is required to auto-download model weights.")
        print("Install it with: pip install huggingface_hub")
        print(f"\nOr manually download from: https://huggingface.co/{HF_REPO_ID}")
        sys.exit(1)

    for hf_name, local_path in missing:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {hf_name} -> {local_path} ...")
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=hf_name,
            local_dir=local_path.parent,
        )
        # hf_hub_download saves with the original filename in local_dir
        downloaded = local_path.parent / hf_name
        if downloaded != local_path and downloaded.exists():
            downloaded.rename(local_path)
        print(f"  Done: {local_path}")


def ensure_weights():
    """Check for weights and download if missing. Called at startup."""
    root = get_project_root()
    missing = check_weights(root)
    if not missing:
        return
    print(f"Missing model weights: {', '.join(m[0] for m in missing)}")
    download_weights(root, missing)
    print("All model weights ready.\n")
