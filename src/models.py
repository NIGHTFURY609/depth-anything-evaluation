"""Thin wrappers around the unmodified upstream DA3 package."""

from typing import Literal

import torch
from depth_anything_3.api import DepthAnything3
from depth_anything_3.model.da3 import DepthAnything3Net
from depth_anything_3.utils.io.input_processor import InputProcessor
from peft import LoraConfig, inject_adapter_in_model
from torch.utils.checkpoint import checkpoint

# Scoped to attention layers: patch_embed.proj also matches a bare "proj".
LORA_TARGET_REGEX = r".*blocks\.\d+\.attn\.(qkv|proj)"


def load_da3(hf_id: str, device: str) -> DepthAnything3:
    return DepthAnything3.from_pretrained(hf_id).to(device).eval()


def preprocess(images: list, process_res: int = 504) -> torch.Tensor:
    """Returns (B, 3, H, W), using the same pipeline as the API's default inference."""
    tensor, _, _ = InputProcessor()(
        images,
        process_res=process_res,
        process_res_method="upper_bound_resize",
        sequential=True,
    )
    return tensor


def forward_da3(net: DepthAnything3Net, x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Forward with gradients enabled, skipping the camera decoder.

    x is (B, 3, H, W). Frames go in as (B, 1, 3, H, W) so that each one is processed alone;
    (1, B, 3, H, W) would make upstream treat them as views of one scene.
    Mirrors the autocast nesting of DepthAnything3.forward and DepthAnything3Net.forward.
    """
    H, W = x.shape[-2:]
    with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16):
        feats, _ = net.backbone(x[:, None], cam_token=None, export_feat_layers=[])
        with torch.autocast(device_type=x.device.type, enabled=False):
            out = net.head(feats, H, W, patch_start_idx=0)
    return {"depth": out.depth[:, 0], "conf": out.depth_conf[:, 0]}


def set_trainable(
    net: DepthAnything3Net,
    mode: Literal["head", "lora", "full"],
    lora_rank: int = 8,
    target_regex: str = LORA_TARGET_REGEX,
) -> None:
    if mode not in ("head", "lora", "full"):
        raise ValueError(f"Unknown fine-tuning mode: {mode}")

    net.requires_grad_(False)
    # The ray branch (*_aux) runs but plays no part in depth, so it never gets gradients.
    for name, param in net.head.named_parameters():
        param.requires_grad_("_aux" not in name)

    if mode == "lora":
        config = LoraConfig(r=lora_rank, lora_alpha=lora_rank, target_modules=target_regex)
        inject_adapter_in_model(config, net.backbone)
    elif mode == "full":
        net.backbone.requires_grad_(True)


def enable_checkpointing(net: DepthAnything3Net) -> None:
    """Upstream's ViT has no gradient checkpointing, so wrap each block's forward."""

    def wrap(forward):
        def checkpointed(*args, **kwargs):
            return checkpoint(forward, *args, use_reentrant=False, **kwargs)

        return checkpointed

    for block in net.backbone.pretrained.blocks:
        block.forward = wrap(block.forward)
