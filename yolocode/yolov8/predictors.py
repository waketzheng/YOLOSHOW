import functools
import json
import os
from pathlib import Path
from typing import Callable, TypeVar

import cv2
import numpy as np
import requests
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.resnet import ResNet
from ultralytics.engine.results import Boxes, Results

T = TypeVar("T")
BASE_DIR = Path(__file__).parent.resolve().parent.parent


class SingleDataset(Dataset):
    def __len__(self) -> int:
        return 1

    def __getitem__(self, index) -> np.float32:
        return self.crop_and_transform(self.im, self.person_box)

    def __init__(self, img: Image.Image | np.ndarray, box: Boxes) -> None:
        if isinstance(img, Image.Image):
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        self.im = img
        self.person_box = box

    @staticmethod
    def square(im: np.ndarray, size=224) -> np.ndarray:
        """把im缩放到尺寸为size的正方形里，多余的部分用0填充"""
        im_bg = np.zeros(shape=(size, size, 3))
        h, w = im.shape[:2]
        if w < h:
            width = int(size * w / h)
            height = size
        else:
            height = int(size * h / w)
            width = size
        im_bg[0:height, 0:width] = cv2.resize(im, dsize=(width, height))
        return im_bg

    @classmethod
    def crop_and_transform(cls, im: np.ndarray, box) -> np.float32:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        new_im = cls.square(im[y1:y2, x1:x2])
        transform = transforms.Compose([transforms.ToTensor()])
        return np.float32(transform(new_im))


def cache_attr(func: Callable[..., T]) -> Callable[..., T]:
    """把函数执行结果缓存到类属性里，方法名若为load_model，则结果会存在`_model`属性

    Usage::
        >>> from ultralytics import YOLO
        >>> class A:
        ...     _model: YOLO  # 这行可省略，但写出来更直观
        ...     @classmethod
        ...     @cache_attr
        ...     def load_model(cls) -> YOLO:
        ...         return YOLO('yolov8n.pt')
    """

    @functools.wraps(func)
    def load_attr(cls, *args, **kw) -> T:
        attr = func.__name__.replace("load", "")
        if (value := getattr(cls, attr, None)) is None:
            value = func(cls, *args, **kw)
            setattr(cls, attr, value)
        return value

    return load_attr


class ColorPredictor:
    default_conf = 0.25
    min_size = 15
    weight_name = "vest_color.pt"
    show_result = True
    # Cached attributions
    _names: dict[int, str]
    _device: str
    _model: ResNet

    @classmethod
    @cache_attr
    def load_names(cls) -> dict[int, str]:
        color_file = BASE_DIR / "config" / "colors.json"
        if not color_file.exists() and (host := os.getenv("YOLOSHOW_HOST")):
            url = host.rstrip("/") + "/" + color_file.name
            content = requests.get(url, verify=False).content
            color_file.write_bytes(content)
        else:
            content = color_file.read_bytes()
        colors: list[str] = json.loads(content)
        return dict(enumerate(colors))

    @classmethod
    @cache_attr
    def load_device(cls) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    @cache_attr
    def load_model(cls) -> ResNet:
        device = cls.load_device()
        model = torchvision.models.resnet18().to(device)
        model.fc = torch.nn.Linear(model.fc.in_features, 11).to(device)
        model_path = BASE_DIR / "ptfiles" / cls.weight_name
        map_location = torch.device(device) if device == "cpu" else None
        model.load_state_dict(torch.load(model_path, map_location=map_location))
        model.eval()
        return model

    @classmethod
    def detect_crop_img(cls, box: Boxes, img) -> tuple[int, float]:
        data = SingleDataset(img, box)
        model = cls.load_model()
        for im in DataLoader(data):
            device = cls.load_device()
            result = model(im.to(device))
            break
        class_id = int(torch.argmax(result, dim=1))
        return class_id, 0.0

    @classmethod
    def update_person_labels(cls, result: Results, img) -> Results:
        if result.boxes is not None:
            default_conf, size = cls.default_conf, cls.min_size
            retain_idx = []
            changed = False
            for i, b in enumerate(result.boxes):
                _, _, w, h = b.xywh[0]
                if w < size or h < size:
                    changed = True
                    continue
                new_class_id, conf = cls.detect_crop_img(b, img)
                if conf:
                    if conf < default_conf:
                        changed = True
                        continue
                    b.conf[0] = conf
                if int(b.cls[0]) != new_class_id:
                    b.cls[0] = new_class_id
                retain_idx.append(i)
            if changed:
                result.update(result.boxes.data[retain_idx])
            result.names = cls.load_names()
            if cls.show_result:
                result.show()
        return result
