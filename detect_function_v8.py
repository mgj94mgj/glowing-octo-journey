import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.ops import xyxy2xywh
from ultralytics.utils.plotting import Annotator

FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))


class YOLOv8Detector:
    """压缩包中的 YOLOv8 检测适配器，保留其三元组输出接口。"""

    def __init__(
        self,
        weights_path,
        img_size=(640, 640),
        conf_thres=0.70,
        iou_thres=0.2,
        max_det=10,
        device='',
        names=None,
        classes=None,
        agnostic_nms=False,
        augment=False,
        visualize=False,
        half=False,
        dnn=False,
        data='data/coco128.yaml',
        ui=False,
    ):
        self.ui = ui
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(weights_path)
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.classes = classes
        self.augment = augment
        self.visualize = visualize
        self.half = half
        self.names = self.model.names if names is None else names
        self.colors = [tuple(random.randint(0, 255) for _ in range(3)) for _ in self.names]

        if self.half and self.device == 'cuda':
            self.model.model.half()

    def predict(self, img):
        im0 = img.copy()
        predictions = self.model.predict(
            source=im0,
            imgsz=self.img_size,
            conf=self.conf_thres,
            iou=self.iou_thres,
            max_det=self.max_det,
            classes=self.classes,
            augment=self.augment,
            verbose=False,
            device=self.device,
        )

        detections = []
        for result in predictions:
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                xywh = xyxy2xywh(torch.tensor(xyxy).view(1, 4)).view(-1).tolist()
                xywh = [round(value) for value in xywh]
                xywh = [
                    xywh[0] - xywh[2] // 2,
                    xywh[1] - xywh[3] // 2,
                    xywh[2],
                    xywh[3],
                ]

                if self.ui:
                    annotator = Annotator(np.ascontiguousarray(img), line_width=3)
                    annotator.box_label(
                        xyxy,
                        f'{self.names[class_id]} {confidence:.2f}',
                        color=self.colors[class_id],
                    )

                detections.append((self.names[class_id], xywh, confidence))
        return detections
