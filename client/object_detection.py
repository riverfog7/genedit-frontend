import os
from os import PathLike
from typing import List, Union

import requests
from PIL import Image

from .models import DetectRequest, DetectorOutput
from .utils import process_image, print_and_raise_for_status


class DetectionClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = os.path.join(base_url, 'detect')
        self.session = requests.Session()

    def detect(
            self,
            image: Union[str, PathLike, bytes, Image.Image],
            text: List[str],
            threshold: float = 0.25
    ) -> DetectorOutput:
        request = DetectRequest(text=text, threshold=threshold)
        files = {"image": process_image(image)}
        data = {"data": request.model_dump_json()}
        response = self.session.post(self.base_url, files=files, data=data)
        print_and_raise_for_status(response)
        return DetectorOutput(**response.json())

    def health_check(self) -> dict:
        response = self.session.get(os.path.join(self.base_url, 'health'))
        print_and_raise_for_status(response)
        return response.json()
