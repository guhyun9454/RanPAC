import os
import math
import hashlib
from typing import List, Optional

import torch
from PIL import Image


def _hash_config(model_id: str, prompts: List[str], negative_prompt: Optional[str], steps: int, guidance: float,
                 height: int, width: int, seed: Optional[int]) -> str:
    m = hashlib.sha256()
    key = "|".join([
        model_id,
        ";".join(prompts),
        negative_prompt or "",
        str(steps), str(guidance), str(height), str(width), str(seed) if seed is not None else "-1"
    ])
    m.update(key.encode("utf-8"))
    return m.hexdigest()[:16]


def default_prompts_for_dataset(dataset_name: str) -> List[str]:
    """Return a small prompt bank for near-OOD synthesis based on the dataset name.

    The goal is to create challenging, realistic near-OODs: visually similar but semantically different.
    """
    name = (dataset_name or "").lower()
    # iDigits: generate letters, symbols, shapes, and textured patterns that resemble strokes
    if "idigits" in name or "digits" in name:
        return [
            "close-up grayscale handwritten letters, high contrast, plain background",
            "vector icons single glyph, monochrome, centered, minimalistic",
            "abstract ink brush strokes, monochrome, on white paper",
            "noisy textured patches, ink splatter, grayscale",
            "random geometric shapes outline drawing, black lines on white",
        ]
    # DomainNet-like (sketch/clipart/painting/real)
    if "domainnet" in name:
        return [
            "abstract clipart icons set on plain background",
            "hand-drawn sketch of random objects, black lines on white",
            "oil painting abstract strokes, colorful, no specific objects",
            "macro textures of fabrics and papers, shallow depth of field",
        ]
    # CIFAR/VTAB/etc: textures, fractals, synthetic objects
    if "cifar" in name or "vtab" in name or "imagenet" in name:
        return [
            "macro photography of textures, seamless pattern, no recognizable objects",
            "3D primitives render: spheres, cubes, cones on plain background",
            "abstract bokeh lights, shallow depth, colorful",
            "fractal patterns, high detail, repeating shapes",
        ]
    # fallback generic
    return [
        "abstract textures and patterns, no recognizable objects",
        "random line drawings and doodles, monochrome on white",
        "vector glyph icons, minimalistic, centered",
    ]


class StableDiffusionOODDataset(torch.utils.data.Dataset):
    """Pre-generate OOD images using a pretrained Stable Diffusion model and serve them as a dataset.

    Images are generated once at initialization and kept in memory. Optionally cached to disk.
    """

    def __init__(
        self,
        num_images: int,
        prompts: List[str],
        *,
        negative_prompt: Optional[str] = None,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        steps: int = 20,
        guidance_scale: float = 7.5,
        height: int = 256,
        width: int = 256,
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
        transform=None,
        cache_dir: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        assert num_images > 0, "num_images must be > 0"
        self.transform = transform
        self.images: List[Image.Image] = []

        # Prepare cache signature
        self._sig = _hash_config(model_id, prompts, negative_prompt, steps, guidance_scale, height, width, seed)
        self._cache_dir = None
        if cache_dir:
            self._cache_dir = os.path.join(cache_dir, f"sd_{self._sig}")
            os.makedirs(self._cache_dir, exist_ok=True)

        # Try to load from cache
        cached = []
        if self._cache_dir and os.path.isdir(self._cache_dir):
            for fname in sorted(os.listdir(self._cache_dir)):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    try:
                        cached.append(Image.open(os.path.join(self._cache_dir, fname)).convert("RGB"))
                    except Exception:
                        pass
        if len(cached) >= num_images:
            self.images = cached[:num_images]
            return

        # Otherwise, generate
        self._generate(
            target=num_images,
            prompts=prompts,
            negative_prompt=negative_prompt,
            model_id=model_id,
            steps=steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            seed=seed,
            device=device,
            dtype=dtype,
        )

        # Save to cache if requested
        if self._cache_dir:
            for idx, img in enumerate(self.images):
                path = os.path.join(self._cache_dir, f"{idx:06d}.png")
                try:
                    img.save(path)
                except Exception:
                    pass

    def _generate(
        self,
        target: int,
        prompts: List[str],
        negative_prompt: Optional[str],
        model_id: str,
        steps: int,
        guidance_scale: float,
        height: int,
        width: int,
        seed: Optional[int],
        device: Optional[torch.device],
        dtype: Optional[torch.dtype],
    ):
        try:
            from diffusers import StableDiffusionPipeline
        except Exception as e:
            raise ImportError(
                "diffusers가 설치되어 있지 않습니다. `pip install diffusers transformers accelerate safetensors`로 설치하세요."
            ) from e

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if dtype is None:
            dtype = torch.float16 if device.type == "cuda" else torch.float32

        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=True)

        # Distribute quota across prompts uniformly
        per_prompt = math.ceil(target / max(1, len(prompts)))
        rnd = torch.Generator(device="cpu")
        if seed is not None:
            rnd.manual_seed(seed)

        for p_idx, prompt in enumerate(prompts):
            remaining = target - len(self.images)
            if remaining <= 0:
                break
            this_n = min(per_prompt, remaining)

            # Generate in small batches for memory
            batch_size = 4 if device.type == "cuda" else 1
            generated = 0
            while generated < this_n:
                b = min(batch_size, this_n - generated)
                g_seed = rnd.seed() if seed is None else seed + p_idx * 100003 + generated
                images = pipe(
                    prompt=[prompt] * b,
                    negative_prompt=[negative_prompt] * b if negative_prompt else None,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                    generator=torch.Generator(device=device).manual_seed(int(g_seed)),
                ).images
                self.images.extend([im.convert("RGB") for im in images])
                generated += b

        # Trim to target
        self.images = self.images[:target]

        # Free GPU memory
        try:
            del pipe
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform is not None:
            img = self.transform(img)
        # label value is ignored by UnknownWrapper in training, but keep a placeholder
        return img, 1


