"""Convert the WASB (BMVC'23) HRNet sports-ball detector to a Core ML ML Program (iOS 15).

Usage: python tools/convert_wasb.py <wasb-repo> <weights.pth.tar> <out.mlpackage>

The exported model takes three BGRA/RGB images (frame0 oldest … frame2 newest, 512x288)
and returns `heatmaps` [1, 3, 288, 512] after sigmoid. ImageNet normalisation is baked in.
"""
import sys

import numpy as np
import torch
import yaml


class AttrDict(dict):
    def __getattr__(self, k):
        v = self[k]
        return AttrDict(v) if isinstance(v, dict) else v


def build(repo, weights):
    sys.path.insert(0, f"{repo}/src")
    from models.hrnet import HRNet

    with open(f"{repo}/src/configs/model/wasb.yaml") as f:
        cfg = AttrDict(yaml.safe_load(f))
    model = HRNet(cfg)
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("missing:", missing, "unexpected:", unexpected)
    if len(missing) > 0:
        raise SystemExit("weights do not match the WASB HRNet config")
    return model.eval()


class Wrapper(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        mean = torch.tensor([0.485, 0.456, 0.406] * 3).view(1, 9, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225] * 3).view(1, 9, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, f0, f1, f2):
        x = torch.cat([f0, f1, f2], dim=1)
        x = (x - self.mean) / self.std
        y = self.net(x)[0]
        return torch.sigmoid(y)


def main():
    repo, weights, out = sys.argv[1:4]
    import coremltools as ct

    net = build(repo, weights)
    w = Wrapper(net).eval()
    ex = [torch.rand(1, 3, 288, 512) for _ in range(3)]
    with torch.no_grad():
        ref = w(*ex).numpy()
        traced = torch.jit.trace(w, ex)
    inputs = [ct.ImageType(name=f"frame{i}", shape=(1, 3, 288, 512), scale=1 / 255.0,
                           color_layout=ct.colorlayout.RGB) for i in range(3)]
    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        outputs=[ct.TensorType(name="heatmaps", dtype=np.float32)],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS15,
        compute_precision=ct.precision.FLOAT16,
    )
    mlmodel.short_description = "WASB HRNet small-ball heatmap detector (tennis weights, MIT licence, nttcom/WASB-SBDT)"
    mlmodel.save(out)
    print("saved", out, "torch heatmap max", float(ref.max()))
    # Sanity check on macOS: compare against PyTorch on the same (image-quantised) input.
    try:
        from PIL import Image
        imgs = [(e[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8) for e in ex]
        pred = mlmodel.predict({f"frame{i}": Image.fromarray(imgs[i]) for i in range(3)})["heatmaps"]
        with torch.no_grad():
            q = [torch.from_numpy(im).permute(2, 0, 1).unsqueeze(0).float() / 255 for im in imgs]
            ref_q = w(*q).numpy()
        print("max abs diff vs torch:", float(np.abs(pred - ref_q).max()))
    except Exception as e:  # prediction only works on macOS
        print("skip predict check:", e)


if __name__ == "__main__":
    main()
