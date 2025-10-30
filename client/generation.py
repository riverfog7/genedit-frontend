import io
import os
from os import PathLike
from typing import Union, Optional

import requests
from PIL import Image

from .models import GenerateRequest, InpaintRequest
from .utils import process_image, print_and_raise_for_status


class ImageGenerationClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = f"{base_url.rstrip('/')}/generate"
        self.session = requests.Session()

    def generate(
            self,
            prompt: str,
            negative_prompt: str = "",
            width: int = 1664,
            height: int = 928,
            num_inference_steps: int = 50,
            true_cfg_scale: float = 4.0,
            seed: Optional[int] = None
    ) -> Image.Image:
        request = GenerateRequest(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=true_cfg_scale,
            seed=seed
        )

        data = {"data": request.model_dump_json()}
        response = self.session.post(f"{self.base_url}/generate", data=data)
        print_and_raise_for_status(response)

        return Image.open(io.BytesIO(response.content))

    def inpaint(
            self,
            control_image: Union[str, PathLike, bytes, Image.Image],
            control_mask: Union[str, PathLike, bytes, Image.Image],
            prompt: str,
            negative_prompt: str = "",
            num_inference_steps: int = 30,
            true_cfg_scale: float = 4.0,
            controlnet_conditioning_scale: float = 1.0,
            seed: Optional[int] = None
    ) -> Image.Image:
        request = InpaintRequest(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=true_cfg_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            seed=seed
        )

        files = {
            "control_image": ("image.png", process_image(control_image), "image/png"),
            "control_mask": ("mask.png", process_image(control_mask), "image/png")
        }
        data = {"data": request.model_dump_json()}

        response = self.session.post(f"{self.base_url}/inpaint", files=files, data=data)
        print_and_raise_for_status(response)

        return Image.open(io.BytesIO(response.content))

    def health_check(self) -> dict:
        response = self.session.get(f"{self.base_url}/health")
        print_and_raise_for_status(response)
        return response.json()
