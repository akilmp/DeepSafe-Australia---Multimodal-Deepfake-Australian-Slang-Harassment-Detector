import argparse
from pathlib import Path

import torch
import torchvision.transforms as T
from torchvision.io import read_video
from torchvision.models import EfficientNet_V2_S_Weights
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from PIL import Image
import imageio

from train_vision import DeepFakeModel


def load_frame(path: str) -> torch.Tensor:
    """Load first frame of a video and apply EfficientNet preprocessing."""
    video, _, _ = read_video(path, pts_unit="sec")
    frame = video[0].permute(2, 0, 1).float() / 255.0
    weights = EfficientNet_V2_S_Weights.DEFAULT
    transform = T.Compose(
        [
            T.Resize(weights.transforms().crop_size),
            T.CenterCrop(weights.transforms().crop_size),
            T.Normalize(mean=weights.meta["mean"], std=weights.meta["std"]),
        ]
    )
    return transform(frame)


def generate_heatmap(model: DeepFakeModel, frame: torch.Tensor) -> Image.Image:
    cam_extractor = GradCAM(model.model, target_layer=model.model.features[-1])
    model.eval()
    with torch.no_grad():
        logits = model(frame.unsqueeze(0))
    cams = cam_extractor(logits.squeeze(0).argmax().item(), logits)
    cam = cams[0]
    inv_norm = T.Normalize(
        mean=[-m / s for m, s in zip(EfficientNet_V2_S_Weights.DEFAULT.meta["mean"], EfficientNet_V2_S_Weights.DEFAULT.meta["std"])],
        std=[1 / s for s in EfficientNet_V2_S_Weights.DEFAULT.meta["std"]],
    )
    img = inv_norm(frame).clamp(0, 1)
    base = T.ToPILImage()(img)
    mask = T.ToPILImage()(cam)
    return overlay_mask(base, mask, alpha=0.5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--ckpt_path", default="best.ckpt")
    args = parser.parse_args()

    model = DeepFakeModel.load_from_checkpoint(args.ckpt_path)
    frame = load_frame(args.video_path)
    result = generate_heatmap(model, frame)

    out_path = Path("app/static/gradcam.gif")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, [result], format="GIF")
    print(f"Saved GradCAM to {out_path}")


if __name__ == "__main__":
    main()
