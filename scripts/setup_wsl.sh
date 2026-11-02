#!/usr/bin/env bash
# Builds the da3ft environment inside WSL. Safe to re-run.
set -euo pipefail

ENV_DIR="$HOME/envs/da3ft"
DA3_DIR="$HOME/src/Depth-Anything-3"
DA3_COMMIT="3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"

export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

[ -x "$ENV_DIR/bin/python" ] || uv venv --python 3.11 "$ENV_DIR"
export VIRTUAL_ENV="$ENV_DIR"

uv pip install torch torchvision --index-url "$TORCH_INDEX"

if [ ! -d "$DA3_DIR/.git" ]; then
  mkdir -p "$(dirname "$DA3_DIR")"
  git clone https://github.com/ByteDance-Seed/Depth-Anything-3 "$DA3_DIR"
fi
git -C "$DA3_DIR" fetch --quiet origin
git -C "$DA3_DIR" checkout --quiet "$DA3_COMMIT"

uv pip install -e "$DA3_DIR"
# addict is imported by depth_anything_3.model.da3 but not declared upstream.
uv pip install addict transformers peft gdown pytest tensorboard scikit-learn pandas matplotlib pyyaml tqdm

"$ENV_DIR/bin/python" -c "import torch, depth_anything_3.api; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.is_bf16_supported())"
