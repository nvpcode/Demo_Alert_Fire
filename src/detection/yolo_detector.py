import os
from typing import List, Dict, Any

import numpy as np
from ultralytics import YOLO


class YoloFireDetector:
    """
    Wrapper đơn giản cho YOLOv8 để phát hiện vùng cháy trên từng frame.

    - Chỉ giữ lại những detection thuộc lớp 'fire' (cấu hình được).
    - Trả về list[dict] với các khóa: bbox, score, class_name.
    """

    def __init__(
        self,
        weights_path: str,
        conf_thres: float = 0.3,
        iou_thres: float = 0.5,
        device: str = "cpu",
        fire_class_names=None,
    ):
        """
        Khởi tạo detector YOLOv8.
        
        Args:
            weights_path: Đường dẫn tới file weights (.pt) của YOLOv8
            conf_thres: Ngưỡng confidence để lọc detection
            iou_thres: Ngưỡng IoU cho NMS
            device: Thiết bị chạy ("cpu" hoặc "cuda")
            fire_class_names: Tên class cần phát hiện. Có thể là:
                - String: "fire" (cho model 1 class)
        """
        if fire_class_names is None:
            fire_class_names = "fire"
        
        # Xử lý cả trường hợp fire_class_names là string hoặc list
        if isinstance(fire_class_names, str):
            fire_class_names = [fire_class_names]
        elif not isinstance(fire_class_names, list):
            fire_class_names = list(fire_class_names)

        # Kiểm tra file model có tồn tại không
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"Không tìm thấy file model tại: {weights_path}\n"
                f"Hãy kiểm tra lại đường dẫn trong config.yaml"
            )

        # Load model YOLOv8
        self.model = YOLO(weights_path)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device
        # Chuyển thành set để so sánh nhanh hơn (không phân biệt hoa thường)
        self.fire_class_names = set([c.lower() for c in fire_class_names])
        
        # Log thông tin model đã load
        print(f"[YoloFireDetector] Đã tải model từ: {weights_path}")
        print(f"[YoloFireDetector] Classes cần phát hiện: {list(self.fire_class_names)}")
        print(f"[YoloFireDetector] Device: {device}, Conf threshold: {conf_thres}, IoU threshold: {iou_thres}")

    def detect_fire(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Chạy YOLOv8 trên một frame và lọc ra vùng cháy.
        """
        # YOLOv8 API: model(frame) trả về list kết quả
        results = self.model.predict(
            source=frame,
            conf=self.conf_thres,
            iou=self.iou_thres,
            device=self.device,
            verbose=False,
        )

        detections: List[Dict[str, Any]] = []

        if not results:
            return detections

        result = results[0]

        boxes = result.boxes  # Boxes object
        if boxes is None or len(boxes) == 0:
            return detections

        for box in boxes:
            cls_id = int(box.cls.item())
            score = float(box.conf.item())

            # Lấy tên lớp
            class_name = result.names.get(cls_id, str(cls_id)).lower()

            # Chỉ giữ những lớp liên quan tới cháy
            if class_name not in self.fire_class_names:
                continue

            xyxy = box.xyxy[0].cpu().numpy().tolist()  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = xyxy

            detections.append(
                {
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "score": score,
                    "class_name": class_name,
                }
            )

        return detections


