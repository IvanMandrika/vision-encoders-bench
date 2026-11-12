import time
from collections.abc import Callable
from typing import TypeVar

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, CLIPVisionModelWithProjection

MODELS = {
    "clip": ("openai/clip-vit-base-patch32", (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    "dinov2": ("facebook/dinov2-small", (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}
OOM_WAIT_SECONDS = 30
T = TypeVar("T")


def retry_on_oom(fn: Callable[[int], T], batch_size: int) -> T:
    while True:
        try:
            return fn(batch_size)
        except torch.OutOfMemoryError:
            if batch_size == 1:
                raise
            torch.cuda.empty_cache()
            batch_size = max(1, batch_size // 2)
            print(f"  CUDA OOM, retrying in {OOM_WAIT_SECONDS} s with batch size {batch_size}", flush=True)
            time.sleep(OOM_WAIT_SECONDS)


class Encoder:
    def __init__(self, name: str, device: str | None = None, size: int = 224):
        repo, mean, std = MODELS[name]
        self.name = name
        self.size = size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        cls = CLIPVisionModelWithProjection if name == "clip" else AutoModel
        self.model = cls.from_pretrained(repo).to(self.device).eval()
        self.mean = torch.tensor(mean, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=self.device).view(1, 3, 1, 1)

    def preprocess(self, images: np.ndarray) -> torch.Tensor:
        x = torch.as_tensor(images, device=self.device).permute(0, 3, 1, 2).float() / 255.0
        if x.shape[-2:] != (self.size, self.size):
            x = F.interpolate(x, size=(self.size, self.size), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
        return (x - self.mean) / self.std

    @torch.no_grad()
    def embed(self, images: np.ndarray, batch_size: int = 128) -> np.ndarray:
        out = []
        for i in range(0, len(images), batch_size):
            x = self.preprocess(images[i : i + batch_size])
            res = self.model(pixel_values=x)
            feat = res.image_embeds if self.name == "clip" else res.pooler_output
            out.append(F.normalize(feat.float(), dim=-1).cpu().numpy())
        return np.concatenate(out)
