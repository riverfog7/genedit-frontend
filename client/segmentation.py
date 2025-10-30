import os
from os import PathLike
from typing import List, Optional, Union

import requests
from PIL import Image

from .models import SegmentationResult, PointSegmentRequest, BoxSegmentRequest, CombinedSegmentRequest
from .utils import process_image, print_and_raise_for_status


class SegmentationClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = os.path.join(base_url, "segment")
        self.session = requests.Session()

    def segment_point(
            self,
            image: Union[str, PathLike, bytes, Image.Image],
            points: List[List[int]],
            labels: List[int]
    ) -> SegmentationResult:
        request = PointSegmentRequest(points=points, labels=labels)
        files = {"image": process_image(image)}
        data = {"data": request.model_dump_json()}
        response = self.session.post(os.path.join(self.base_url, "point"), files=files, data=data)
        print_and_raise_for_status(response)
        return SegmentationResult(response.content)

    def segment_box(
            self,
            image: Union[str, PathLike, bytes, Image.Image],
            box: List[int]
    ) -> SegmentationResult:
        request = BoxSegmentRequest(box=box)
        files = {"image": process_image(image)}
        data = {"data": request.model_dump_json()}
        response = self.session.post(os.path.join(self.base_url, "box"), files=files, data=data)
        print_and_raise_for_status(response)
        return SegmentationResult(response.content)

    def segment_combined(
            self,
            image: Union[str, PathLike, bytes, Image.Image],
            points: Optional[List[List[int]]] = None,
            labels: Optional[List[int]] = None,
            box: Optional[List[int]] = None
    ) -> SegmentationResult:
        request = CombinedSegmentRequest(points=points, labels=labels, box=box)
        files = {"image": process_image(image)}
        data = {"data": request.model_dump_json()}
        response = self.session.post(os.path.join(self.base_url, "combined"), files=files, data=data)
        print_and_raise_for_status(response)
        return SegmentationResult(response.content)

    def health_check(self) -> dict:
        response = self.session.get(os.path.join(self.base_url, "health"))
        print_and_raise_for_status(response)
        return response.json()
