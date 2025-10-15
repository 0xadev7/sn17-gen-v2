from typing import Tuple
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation


class BiRefNetRemover:
    def __init__(self, device: torch.device):
        self.device = device
        self.model = (
            AutoModelForImageSegmentation.from_pretrained(
                "ZhengPeng7/BiRefNet", trust_remote_code=True
            )
            .to(self.device)
            .eval()
        )

        if self.device.type == "cuda":
            self.model.half()
            torch.set_float32_matmul_precision("high")

        self.image_size = (1024, 1024)
        self.tfm = transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    @torch.inference_mode()
    def remove(self, img: Image.Image) -> Tuple[Image.Image, np.ndarray]:
        rgb = img.convert("RGB")

        # Your mask prediction (HxW in [0,1])
        x = self.tfm(rgb).unsqueeze(0).to(self.device, non_blocking=True)
        if self.device.type == "cuda":
            x = x.half()

        pred = self.model(x)[-1].sigmoid().float().cpu()[0, 0]  # HxW, float32 in [0,1]

        # Resize mask back to image size and convert to 8-bit alpha
        mask_pil = transforms.functional.resize(
            transforms.ToPILImage()(pred), rgb.size, antialias=True
        )
        alpha = np.array(mask_pil, dtype=np.uint8)  # 0..255

        # Attach alpha -> RGBA
        rgba = rgb.copy()
        rgba.putalpha(Image.fromarray(alpha))  # <- now 4-channel

        return rgba
