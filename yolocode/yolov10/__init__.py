# Ultralytics YOLO 🚀, AGPL-3.0 license

__version__ = "8.1.34"

from ultralytics.data.explorer.explorer import Explorer
from ultralytics.models.fastsam import FastSAM
from ultralytics.models.nas import NAS
from ultralytics.utils import ASSETS
from ultralytics.utils import SETTINGS as settings
from ultralytics.utils.checks import check_yolo as checks
from ultralytics.utils.downloads import download

from yolocode.yolov10.models import RTDETR, SAM, YOLO, YOLOv10, YOLOWorld

__all__ = (
    "__version__",
    "ASSETS",
    "YOLO",
    "YOLOWorld",
    "NAS",
    "SAM",
    "FastSAM",
    "RTDETR",
    "checks",
    "download",
    "settings",
    "Explorer",
    "YOLOv10",
)
