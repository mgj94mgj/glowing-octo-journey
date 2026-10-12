# 导入需要的库
import os
import sys
import platform
import time
import importlib.util
from pathlib import Path, PurePosixPath, WindowsPath

# Windows下将PosixPath映射到WindowsPath，解决Linux保存的.pt模型在Windows加载报错的问题
import pathlib
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath

# 初始化目录
import cv2

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # 定义YOLOv5的根目录
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # 将YOLOv5的根目录添加到环境变量中（程序结束后删除）
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.general import (LOGGER, check_img_size, scale_boxes)

# 导入letterbox

import random
import torch
import numpy as np
from utils.general import non_max_suppression, xyxy2xywh
from utils.torch_utils import select_device
from utils.plots import Annotator
from models.common import DetectMultiBackend
from utils.augmentations import letterbox


def select_detection_output(pred, num_classes=None):
    """从 TensorRT 多输出列表中取 YOLO 检测张量。

    某些由 ONNX 转换得到的 engine 会暴露检测头的中间输出；真正的
    YOLOv8/11 检测输出形状为 [batch, 4 + num_classes, anchors]，后续
    需要将它转置并交给统一的 NMS。
    """
    if not isinstance(pred, (list, tuple)):
        return pred

    candidates = [
        output for output in pred
        if isinstance(output, torch.Tensor)
        and output.ndim == 3
        and output.shape[1] >= 5
        and output.shape[1] < output.shape[2]
    ]
    # TensorRT 不保证多输出 binding 的返回顺序。优先用类别数匹配检测头的
    # 通道数：YOLOv8/11 为 4+nc，部分导出图会保留为 5+nc。
    if num_classes is not None:
        expected_channels = {int(num_classes) + 4, int(num_classes) + 5}
        matched = [output for output in candidates if output.shape[1] in expected_channels]
        if matched:
            candidates = matched
    if not candidates:
        shapes = [tuple(output.shape) for output in pred if hasattr(output, 'shape')]
        raise RuntimeError(f"TensorRT 输出中未找到 YOLO 检测张量，输出形状: {shapes}")
    return candidates[0]


class YOLOv5Detector:
    def __init__(self, weights_path, img_size=(640, 640), conf_thres=0.70, iou_thres=0.2, max_det=10,
                 device='', classes=None, agnostic_nms=False, augment=False, visualize=False, half=True, dnn=False,
                 data='data/coco128.yaml', ui=False):
        # 设置设备：默认遵从上层传入，必要时安全回退到CPU
        self.ui = ui
        requested_device = str(device).strip().lower() if device is not None else ''
        weights_path = str(weights_path)

        if weights_path.lower().endswith('.engine') and not torch.cuda.is_available():
            raise RuntimeError(
                f'TensorRT engine 只能在可用 CUDA 驱动的 GPU 上运行，当前 CUDA 不可用：{weights_path}'
            )

        if requested_device in ('', 'none'):
            # select_device('cpu') 会设置 CUDA_VISIBLE_DEVICES=-1，导致后续 TensorRT
            # 无法再初始化 CUDA（.engine 会段错误），故 TensorRT 权重默认走 GPU
            wlow = str(weights_path).lower()
            if wlow.endswith('.engine'):
                requested_device = '0'
            else:
                requested_device = 'cpu'

        # YOLOv5 的 select_device 常用 '0' 指代第一张GPU
        if requested_device.startswith('cuda'):
            if ':' in requested_device:
                requested_device = requested_device.split(':', 1)[1] or '0'
            else:
                requested_device = '0'
        elif requested_device == 'gpu':
            requested_device = '0'

        # CUDA不可用时回退CPU，避免初始化报错
        if requested_device != 'cpu' and not torch.cuda.is_available():
            requested_device = 'cpu'

        self.device = select_device(requested_device)
        print(f"使用设备: {self.device} (requested={device})")

        if str(weights_path).lower().endswith('.onnx') and not dnn:
            if importlib.util.find_spec('onnxruntime') is None:
                dnn = True
                print("未安装 onnxruntime，自动使用 OpenCV DNN 加载 ONNX")

        # 加载模型 - 禁用半精度(CPU不支持FP16)
        self.model = DetectMultiBackend(weights_path, device=self.device, dnn=dnn, fp16=False, data=data)

        stride, self.names, pt, jit, onnx, engine = self.model.stride, self.model.names, self.model.pt, self.model.jit, self.model.onnx, self.model.engine
        class_aliases = {'Bsentry': 'B7', 'Rsentry': 'R7'}
        if isinstance(self.names, dict):
            self.names = {index: class_aliases.get(name, name) for index, name in self.names.items()}
        else:
            self.names = [class_aliases.get(name, name) for name in self.names]
        if engine and 'images' in self.model.bindings:
            engine_input_shape = self.model.bindings['images'].shape
            img_size = tuple(engine_input_shape[-2:])
            print(f"TensorRT 输入尺寸: {img_size}")
        self.img_size = check_img_size(img_size, s=stride)
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
        # CPU模式下禁用半精度
        self.half = False  # CPU不支持半精度
        if pt or jit:
            self.model.model.float()  # 使用全精度Float32
        self.save_time = 0
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.augment = augment
        self.visualize = visualize
        bs = 1  # batch_size
        # 开始预测
        self.model.warmup(imgsz=(1 if pt or self.model.triton else bs, 3, *self.img_size))  # warmup

    def predict(self, img):
        # 对图片进行处理

        im0 = img.copy()
        im = letterbox(im0, self.img_size, self.model.stride, auto=self.model.pt)[0]
        # cv2.imshow('im', im)
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)

        im = torch.from_numpy(im).to(self.device)
        im = im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # 预测
        pred = self.model(im, augment=self.augment, visualize=self.visualize)
        pred = select_detection_output(pred, num_classes=len(self.names))

        # Ultralytics YOLOv8/11 ONNX 输出为 [B, 4+nc, N]，转成 YOLOv5 NMS
        # 使用的 [B, N, 5+nc]，其中新增的 objectness 固定为 1。
        if pred.ndim == 3 and pred.shape[1] < pred.shape[2] and pred.shape[1] >= 5:
            pred = pred.transpose(1, 2)
            objectness = torch.ones(
                (*pred.shape[:2], 1),
                dtype=pred.dtype,
                device=pred.device
            )
            pred = torch.cat((pred[..., :4], objectness, pred[..., 4:]), dim=2)

        # NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms,
                                   max_det=self.max_det)

        # 用于存放结果
        detections = []

        # 处理预测结果
        for i, det in enumerate(pred):
            # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):

                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                # print(det)
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
                    xywh = [round(x) for x in xywh]
                    xywh = [xywh[0] - xywh[2] // 2, xywh[1] - xywh[3] // 2, xywh[2], xywh[3]]
                    if self.ui:
                        annotator = Annotator(np.ascontiguousarray(img), line_width=3, example=str(self.names))
                        # print(int(cls))
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        annotator.box_label(xyxy, label, color=self.colors[int(cls)])

                    cls = self.names[int(cls)]
                    conf = float(conf)
                    line = (cls, xywh, conf)
                    detections.append(line)
        # if self.save_data:
        #     if time.time() - self.save_time >= 1:
        #         self.save_time = time.time()
        #         save_path = "save_data/img/" + str(round(time.time())) + ".jpg"
        #         cv2.imwrite(save_path, img)
        #         # save_path = "save_data/img_ui/img_ui_" + str(round(time.time())) + ".jpg"
        #         # cv2.imwrite(save_path, im0)
                # print(1)

        # LOGGER.info(f'({t3 - t2:.3f}s)')

        return detections
