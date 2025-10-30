from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import Union, Optional

from PIL import Image, ImageDraw, ImageFont

from .models import DetectorOutput


def process_image(
        image: Union[str, PathLike, bytes, Image.Image],
        max_size: int = 50 * 1024 * 1024
) -> BytesIO:
    if isinstance(image, bytes):
        if len(image) > max_size:
            raise ValueError(f"Image size exceeds {max_size} bytes")
        return BytesIO(image)

    elif isinstance(image, (str, PathLike)):
        path = Path(image)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        if path.stat().st_size > max_size:
            raise ValueError(f"Image file too large: {path.stat().st_size} bytes")

        with open(path, "rb") as f:
            return BytesIO(f.read())

    elif isinstance(image, Image.Image):
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        return buffer

    else:
        raise TypeError(f"Unsupported image type: {type(image)}")


def draw_detections(
        image: Image.Image,
        detections: DetectorOutput,
        box_color: str = "red",
        text_color: str = "white",
        box_width: int = 3,
        font_size: Optional[int] = None
) -> Image.Image:
    image = image.copy()
    draw = ImageDraw.Draw(image)

    if font_size is None:
        font_size = max(12, int(min(image.width, image.height) * 0.02))

    font = ImageFont.load_default()

    for detection in detections.detections:
        for box, score, label in zip(detection.boxes, detection.scores, detection.labels):
            x1, y1, x2, y2 = box

            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=box_width)

            text = f"{label}: {score:.2f}"
            bbox = draw.textbbox((x1, y1), text, font=font)
            text_height = bbox[3] - bbox[1]

            draw.rectangle([x1, y1 - text_height - 4, x1 + bbox[2] - bbox[0] + 8, y1], fill=box_color)
            draw.text((x1 + 4, y1 - text_height - 2), text, fill=text_color, font=font)

    return image


def print_and_raise_for_status(response):
    try:
        response.raise_for_status()
    except Exception as e:
        print(f"Request failed: {e}")
        print(f"Response content: {response.content.decode('utf-8')}")
        raise e
