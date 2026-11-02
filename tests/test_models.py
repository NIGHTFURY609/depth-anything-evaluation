from pathlib import Path

import numpy as np
import pytest
import torch

from src.config import load_config
from src.models import enable_checkpointing, forward_da3, load_da3, preprocess, set_trainable

pytestmark = pytest.mark.gpu

HF_ID = "depth-anything/DA3-SMALL"
DEVICE = "cuda"


def noise_image(seed: int) -> np.ndarray:
    """Smooth 1920x1200 image; pure noise gives the model nothing to separate frames on."""
    rng = np.random.default_rng(seed)
    coarse = rng.integers(0, 255, (12, 20, 3), dtype=np.uint8)
    return np.kron(coarse, np.ones((100, 96, 1), dtype=np.uint8))


@pytest.fixture(scope="module")
def api():
    return load_da3(HF_ID, DEVICE)


def rel_diff(a, b) -> float:
    a, b = torch.as_tensor(a, dtype=torch.float32), torch.as_tensor(b, dtype=torch.float32)
    return ((a - b).abs() / b.abs().clamp_min(1e-6)).max().item()


def test_output_size(api):
    x = preprocess([noise_image(0)]).to(DEVICE)
    with torch.no_grad():
        out = forward_da3(api.model, x)
    assert out["depth"].shape == (1, 322, 504)
    assert out["conf"].shape == (1, 322, 504)


def test_forward_matches_api(api):
    image = noise_image(1)
    pred = api.inference([image], process_res=504, process_res_method="upper_bound_resize")
    x = preprocess([image]).to(DEVICE)
    with torch.no_grad():
        out = forward_da3(api.model, x)
    assert rel_diff(out["depth"][0].cpu(), pred.depth[0]) < 1e-4
    assert rel_diff(out["conf"][0].cpu(), pred.conf[0]) < 1e-4


def test_batch_is_independent(api):
    images = [noise_image(2), noise_image(3)]
    x = preprocess(images).to(DEVICE)
    with torch.no_grad():
        batched = forward_da3(api.model, x)["depth"]
        single = [forward_da3(api.model, x[i : i + 1])["depth"][0] for i in range(2)]
    assert rel_diff(batched[0], single[0]) < 1e-2
    assert rel_diff(batched[1], single[1]) < 1e-2


def test_multiview_differs(api):
    x = preprocess([noise_image(2), noise_image(3)]).to(DEVICE)
    with torch.no_grad():
        batched = forward_da3(api.model, x)["depth"]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            multiview = api.model(x[None]).depth[0]
    assert rel_diff(multiview, batched) > 1e-2


def test_gradients_reach_depth_and_conf(api):
    x = preprocess([noise_image(4)]).to(DEVICE)
    block = api.model.backbone.pretrained.blocks[0].attn.proj.weight
    block.requires_grad_(True)
    block.grad = None
    out = forward_da3(api.model, x)
    assert out["depth"].requires_grad and out["conf"].requires_grad
    (out["depth"].mean() + out["conf"].mean()).backward()
    assert block.grad is not None and block.grad.abs().sum() > 0
    api.model.zero_grad(set_to_none=True)


def test_conf_floor(api):
    x = preprocess([noise_image(5)]).to(DEVICE)
    with torch.no_grad():
        assert forward_da3(api.model, x)["conf"].min() >= 1


REAL_FRAMES = sorted(
    str(next(p.glob("pylon_camera_node/frame*.jpg")))
    for p in (Path(load_config()["paths"]["data_root"]) / "Rellis-3D").glob("0000?")
    if any(p.glob("pylon_camera_node/frame*.jpg"))
)


@pytest.mark.skipif(not REAL_FRAMES, reason="RELLIS images not extracted yet")
@pytest.mark.parametrize("path", REAL_FRAMES)
def test_forward_matches_api_on_real_frame(api, path):
    pred = api.inference([path], process_res=504, process_res_method="upper_bound_resize")
    x = preprocess([path]).to(DEVICE)
    with torch.no_grad():
        out = forward_da3(api.model, x)
    assert out["depth"].shape == (1, 322, 504)
    assert rel_diff(out["depth"][0].cpu(), pred.depth[0]) < 1e-4
    assert rel_diff(out["conf"][0].cpu(), pred.conf[0]) < 1e-4


# A fresh network per test: these tests change requires_grad and inject LoRA layers.
@pytest.fixture
def net():
    model = load_da3(HF_ID, DEVICE).model
    yield model
    del model
    torch.cuda.empty_cache()


def backward_once(net) -> None:
    x = preprocess([noise_image(6)]).to(DEVICE)
    out = forward_da3(net, x)
    (out["depth"].mean() + out["conf"].mean()).backward()


def test_lora_param_count(net):
    set_trainable(net, "lora", lora_rank=8)
    count = sum(p.numel() for n, p in net.named_parameters() if "lora_" in n)
    assert count == 27648 * 8


def test_lora_skips_patch_embed(net):
    set_trainable(net, "lora", lora_rank=8)
    assert not [n for n, _ in net.named_parameters() if "lora_" in n and "patch_embed" in n]


@pytest.mark.parametrize("mode", ["head", "lora", "full"])
def test_every_trainable_param_gets_grad(net, mode):
    set_trainable(net, mode)
    backward_once(net)
    missing = [n for n, p in net.named_parameters() if p.requires_grad and p.grad is None]
    assert missing == []


@pytest.mark.parametrize("mode", ["head", "lora", "full"])
def test_camera_modules_stay_frozen(net, mode):
    set_trainable(net, mode)
    assert not [n for n, p in net.cam_dec.named_parameters() if p.requires_grad]
    assert not [n for n, p in net.cam_enc.named_parameters() if p.requires_grad]


def test_frozen_params_unchanged(net):
    set_trainable(net, "head")
    frozen = net.backbone.pretrained.blocks[0].attn.proj.weight
    trained = net.head.scratch.output_conv2[-1].weight
    frozen_before, trained_before = frozen.clone(), trained.clone()
    optimizer = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=1e-3)
    backward_once(net)
    optimizer.step()
    assert torch.equal(frozen, frozen_before)
    assert not torch.equal(trained, trained_before)


def test_lora_starts_identical(net):
    x = preprocess([noise_image(7)]).to(DEVICE)
    with torch.no_grad():
        before = forward_da3(net, x)["depth"].clone()
    set_trainable(net, "lora")
    with torch.no_grad():
        after = forward_da3(net, x)["depth"]
    assert rel_diff(after, before) < 1e-5


def test_lora_changes_output_after_step(net):
    x = preprocess([noise_image(7)]).to(DEVICE)
    set_trainable(net, "lora")
    with torch.no_grad():
        before = forward_da3(net, x)["depth"].clone()
    optimizer = torch.optim.AdamW(
        [p for n, p in net.named_parameters() if "lora_" in n], lr=1e-3
    )
    backward_once(net)
    optimizer.step()
    with torch.no_grad():
        after = forward_da3(net, x)["depth"]
    assert rel_diff(after, before) > 1e-5


def test_checkpointing_same_output(net):
    x = preprocess([noise_image(8)]).to(DEVICE)
    set_trainable(net, "full")
    plain = forward_da3(net, x)["depth"].detach().clone()
    enable_checkpointing(net)
    out = forward_da3(net, x)
    assert rel_diff(out["depth"].detach(), plain) < 1e-4
    out["depth"].mean().backward()
    assert net.backbone.pretrained.blocks[0].attn.proj.weight.grad is not None
