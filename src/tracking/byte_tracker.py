"""
ByteTrack Tracker - Wrapper cho ByteTrack từ Ultralytics.
Sử dụng Kalman Filter và data association nâng cao để tracking chính xác hơn.
Cấu hình dựa trên bytetrack.yaml
"""

from typing import List, Dict, Any
import numpy as np
from ultralytics.trackers import BYTETracker
from ultralytics.utils import IterableSimpleNamespace


class ResultsProxy:
    """
    Proxy object để giả lập Results object mà ByteTracker cần.
    ByteTracker yêu cầu object có:
    - Attribute 'conf' (confidence scores)
    - Attribute 'boxes' hoặc có thể index được như numpy array
    - Attribute 'xywh' (center x, center y, width, height)
    - Có thể slice/index: results[inds]
    """
    
    def __init__(self, boxes, scores, classes):
        """
        Args:
            boxes: numpy array shape (N, 4) - [x1, y1, x2, y2] (xyxy format)
            scores: numpy array shape (N,) - confidence scores
            classes: numpy array shape (N,) - class IDs
        """
        self.boxes = boxes  # [x1, y1, x2, y2] - xyxy format
        self.conf = scores  # Confidence scores - ByteTracker cần attribute này
        self.cls = classes  # Class IDs
        
        # Tạo data array để có thể index được như results[inds]
        # Format: [x1, y1, x2, y2, conf, cls]
        if len(boxes) > 0:
            self.data = np.column_stack([boxes, scores, classes])
        else:
            self.data = np.empty((0, 6), dtype=np.float32)
        self._len = len(boxes)
    
    @property
    def xywh(self) -> np.ndarray:
        """
        Chuyển đổi từ xyxy format sang xywh format.
        xywh = [x_center, y_center, width, height]
        
        Returns:
            numpy array shape (N, 4) với format [x_center, y_center, width, height]
        """
        if len(self.boxes) == 0:
            return np.empty((0, 4), dtype=np.float32)
        
        # Chuyển đổi từ [x1, y1, x2, y2] sang [x_center, y_center, width, height]
        x1, y1, x2, y2 = self.boxes[:, 0], self.boxes[:, 1], self.boxes[:, 2], self.boxes[:, 3]
        x_center = (x1 + x2) / 2.0
        y_center = (y1 + y2) / 2.0
        width = x2 - x1
        height = y2 - y1
        
        return np.column_stack([x_center, y_center, width, height]).astype(np.float32)
    
    def __len__(self):
        return self._len
    
    def __getitem__(self, indices):
        """Cho phép index như results[inds] - ByteTracker cần tính năng này"""
        if isinstance(indices, (int, np.integer)):
            # Single index
            return ResultsProxy(
                self.boxes[indices:indices+1],
                self.conf[indices:indices+1],
                self.cls[indices:indices+1]
            )
        else:
            # Slice hoặc boolean array
            return ResultsProxy(
                self.boxes[indices],
                self.conf[indices],
                self.cls[indices]
            )


class ByteTrackerWrapper:
    """
    Wrapper cho ByteTrack tracker từ Ultralytics.
    
    ByteTrack sử dụng:
    - Kalman Filter để dự đoán vị trí object
    - Data association 2 bước (high score + low score detections)
    - Quản lý track state (tracked, lost, removed)
    
    Cấu hình dựa trên bytetrack.yaml:
    - track_high_thresh: 0.25 (ngưỡng cho detections chất lượng cao)
    - track_low_thresh: 0.1 (ngưỡng cho detections chất lượng thấp)
    - new_track_thresh: 0.25 (ngưỡng để tạo track mới)
    - track_buffer: 30 (số frame giữ track bị lost)
    - match_thresh: 0.8 (ngưỡng IoU để match)
    - fuse_score: True (kết hợp score với motion/IoU)
    """

    def __init__(
        self,
        track_high_thresh: float = 0.25,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.25,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        frame_rate: int = 30,
    ):
        """
        Khởi tạo ByteTracker với các tham số từ bytetrack.yaml.
        
        Args:
            track_high_thresh: Ngưỡng confidence cho detections chất lượng cao (mặc định: 0.25)
            track_low_thresh: Ngưỡng confidence cho detections chất lượng thấp (mặc định: 0.1)
            new_track_thresh: Ngưỡng để tạo track mới (mặc định: 0.25)
            track_buffer: Số frame tối đa track có thể bị lost trước khi xóa (mặc định: 30)
            match_thresh: Ngưỡng IoU để match track với detection (mặc định: 0.8)
            frame_rate: Frame rate của video (mặc định: 30)
        """
        # Tạo args object cho ByteTracker theo đúng format bytetrack.yaml
        args = IterableSimpleNamespace(
            track_thresh=new_track_thresh,  # Dùng new_track_thresh cho track_thresh
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            track_high_thresh=track_high_thresh,
            track_low_thresh=track_low_thresh,
            new_track_thresh=new_track_thresh,
            proximity_thresh=0.5,
            appearance_thresh=0.25,
            with_reid=False,
            fuse_score=True,  # Kết hợp score với motion/IoU
        )
        
        self.tracker = BYTETracker(args, frame_rate=frame_rate)
        self.frame_id = 0

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Cập nhật tracker với detections mới.
        
        Args:
            detections: List[dict] với keys: bbox [x1,y1,x2,y2], score, class_name
            
        Returns:
            List[dict] với keys: id, bbox, score, class_name="fire", lost, state
            - bbox: [x1, y1, x2, y2] - tọa độ bounding box
            - id: track ID duy nhất
            - class_name: "fire" - nhãn class
            - score: confidence score (0.0 - 1.0)
        """
        self.frame_id += 1
        
        # Chuyển đổi detections sang format mà ByteTracker cần
        if len(detections) == 0:
            # Không có detection => tạo ResultsProxy rỗng
            empty_boxes = np.empty((0, 4), dtype=np.float32)
            empty_scores = np.empty((0,), dtype=np.float32)
            empty_classes = np.empty((0,), dtype=np.int32)
            results_obj = ResultsProxy(empty_boxes, empty_scores, empty_classes)
        else:
            # Có detections => chuyển đổi sang numpy arrays
            boxes = []
            scores = []
            classes = []
            
            for det in detections:
                boxes.append(det["bbox"])  # [x1, y1, x2, y2]
                scores.append(det["score"])
                # ByteTracker cần class_id (int), model chỉ có 1 class "fire" nên dùng 0
                classes.append(0)
            
            boxes = np.array(boxes, dtype=np.float32)
            scores = np.array(scores, dtype=np.float32)
            classes = np.array(classes, dtype=np.int32)
            
            # Tạo ResultsProxy object
            results_obj = ResultsProxy(boxes, scores, classes)
        
        # Cập nhật tracker
        try:
            tracked_results = self.tracker.update(results_obj, img=None)
        except Exception as e:
            # Nếu có lỗi, log và trả về danh sách rỗng
            print(f"Lỗi trong ByteTracker.update(): {e}")
            import traceback
            traceback.print_exc()
            return []
        
        return self._convert_to_output_format(tracked_results)

    def _convert_to_output_format(self, tracked_results: np.ndarray) -> List[Dict[str, Any]]:
        """
        Chuyển đổi kết quả từ ByteTracker sang format chuẩn của hệ thống.
        
        ByteTracker trả về numpy array với format: [x1, y1, x2, y2, track_id, score]
        
        Args:
            tracked_results: numpy array từ ByteTracker với shape (N, 6)
                            
        Returns:
            List[dict] với format đầy đủ:
            - id: track ID (int)
            - bbox: [x1, y1, x2, y2] (list[float])
            - class_name: "fire" (str)
            - score: confidence score (float)
            - lost: số frame bị lost (int)
            - state: "tracked" | "lost" | "unknown" (str)
        """
        outputs = []
        
        if tracked_results is None or len(tracked_results) == 0:
            return outputs
        
        # Lấy danh sách track IDs hiện tại để xác định state
        tracked_ids = {strack.track_id for strack in self.tracker.tracked_stracks}
        lost_ids = {strack.track_id for strack in self.tracker.lost_stracks}
        
        for result in tracked_results:
            if len(result) < 6:
                continue
                
            x1, y1, x2, y2, track_id, score = result[:6]
            track_id = int(track_id)
            score = float(score)
            
            # Xác định state của track
            if track_id in tracked_ids:
                state = "tracked"
                lost = 0
            elif track_id in lost_ids:
                state = "lost"
                lost = 1
            else:
                state = "unknown"
                lost = 0
            
            # Tạo output với đầy đủ thông tin: bbox, id, nhãn "fire", confidence
            outputs.append({
                "id": track_id,                                    # Track ID
                "bbox": [float(x1), float(y1), float(x2), float(y2)],  # Bounding box
                "class_name": "fire",                             # Nhãn class
                "score": score,                                    # Confidence score
                "lost": lost,                                      # Số frame bị lost
                "state": state,                                    # Trạng thái track
            })
        
        return outputs
