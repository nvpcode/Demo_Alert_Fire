from typing import List, Dict, Any

import cv2
import numpy as np


def draw_detections_and_tracks(
    frame, 
    detections: List[Dict[str, Any]] = None,
    tracks: List[Dict[str, Any]] = None, 
    show_ids: bool = True,
    show_detections: bool = True,
    show_tracks: bool = True,
):
    """
    Vẽ bounding box cho cả detections (trước tracking) và tracks (sau tracking) lên frame.
    
    Args:
        frame: Ảnh gốc (numpy array)
        detections: List detections từ YOLO (chưa tracking) - vẽ màu xanh dương
        tracks: List tracks sau khi tracking - vẽ màu xanh lá
        show_ids: Hiển thị ID track
        show_detections: Có vẽ detections không
        show_tracks: Có vẽ tracks không
    """
    vis = frame.copy()

    # Vẽ DETECTIONS (màu xanh dương - cyan) - tất cả các lửa model phát hiện được
    if show_detections and detections:
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            score = det.get("score", 0.0)
            class_name = det.get("class_name", "fire")
            
            # Màu xanh dương cho detections
            color = (255, 255, 0)  # Cyan (BGR)
            thickness = 2
            
            # Vẽ bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
            
            # Vẽ label với confidence
            label = f"{class_name} {score:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            
            # Vẽ background cho text
            cv2.rectangle(
                vis,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1,
            )
            
            # Vẽ text
            cv2.putText(
                vis,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),  # Màu đen cho text
                2,
                cv2.LINE_AA,
            )

    # Vẽ TRACKS (màu xanh lá) - các lửa đang được tracking
    if show_tracks and tracks:
        for tr in tracks:
            x1, y1, x2, y2 = map(int, tr["bbox"])
            tid = tr["id"]
            score = tr.get("score", 0.0)
            class_name = tr.get("class_name", "fire")
            lost = tr.get("lost", 0)
            state = tr.get("state", "tracked")

            # Màu xanh lá cho tracks đang active, vàng cho tracks bị lost
            if lost == 0 and state == "tracked":
                color = (0, 255, 0)  # Xanh lá
                thickness = 3
            else:
                color = (0, 255, 255)  # Vàng (lost)
                thickness = 2

            # Vẽ bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
            
            # Tạo label với ID và confidence
            if show_ids:
                label = f"ID:{tid} {score:.2f}"
                if lost > 0:
                    label += f" (lost)"
            else:
                label = f"{class_name} {score:.2f}"
            
            # Vẽ background cho text
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                vis,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1,
            )
            
            # Vẽ text
            cv2.putText(
                vis,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),  # Màu đen cho text
                2,
                cv2.LINE_AA,
            )
    
    return vis


