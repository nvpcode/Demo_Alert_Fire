from typing import List, Dict, Any

import numpy as np


def compute_iou(boxA, boxB) -> float:
    """
    Tính IoU giữa hai bounding box (x1, y1, x2, y2).
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    if inter_area <= 0:
        return 0.0

    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = inter_area / float(boxA_area + boxB_area - inter_area + 1e-6)
    return float(iou)


class Track:
    """
    Một track đơn lẻ, đại diện cho một vùng cháy được theo dõi qua thời gian.
    """

    def __init__(self, track_id: int, bbox, score: float, class_name: str):
        self.id = track_id
        self.bbox = bbox
        self.score = score
        self.class_name = class_name
        self.lost = 0  # số frame liên tiếp không match detection mới


class IoUTracker:
    """
    Tracker đơn giản dựa trên IoU:

    - Mỗi frame, ghép detection mới với track cũ dựa trên IoU lớn nhất.
    - Nếu IoU > ngưỡng, cập nhật track tương ứng.
    - Detection không match => tạo track mới.
    - Track không match detection mới => tăng bộ đếm 'lost'.
    - Nếu 'lost' > max_lost => xóa track.

    Lưu ý: Đây là phiên bản MVP, đơn giản hơn nhiều so với ByteTrack thực sự,
    nhưng vẫn minh họa được ý tưởng theo dõi ID qua nhiều khung hình.
    """

    def __init__(self, iou_threshold: float = 0.3, max_lost: int = 10):
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.tracks: Dict[int, Track] = {}
        self.next_id: int = 1

    def _create_track(self, det: Dict[str, Any]) -> Track:
        t = Track(
            track_id=self.next_id,
            bbox=det["bbox"],
            score=det["score"],
            class_name=det["class_name"],
        )
        self.tracks[self.next_id] = t
        self.next_id += 1
        return t

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Cập nhật tracker với danh sách detection ở frame hiện tại.

        Trả về list[dict]:
        - id, bbox, score, class_name, lost
        """
        if len(self.tracks) == 0:
            # Chưa có track nào => tạo track cho tất cả detection
            for det in detections:
                self._create_track(det)
        else:
            # Ma trận IoU giữa track và detection
            track_ids = list(self.tracks.keys())
            track_bboxes = [self.tracks[tid].bbox for tid in track_ids]

            if len(detections) > 0:
                det_bboxes = [d["bbox"] for d in detections]
                iou_matrix = np.zeros((len(track_bboxes), len(det_bboxes)), dtype=float)

                for i, tb in enumerate(track_bboxes):
                    for j, db in enumerate(det_bboxes):
                        iou_matrix[i, j] = compute_iou(tb, db)

                # Gán greedily: mỗi detection & track chỉ match 1 lần
                used_tracks = set()
                used_dets = set()

                # Lặp cho đến khi không còn cặp IoU nào vượt ngưỡng
                while True:
                    max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                    max_iou = iou_matrix[max_idx]
                    if max_iou < self.iou_threshold:
                        break

                    t_idx, d_idx = max_idx
                    if t_idx in used_tracks or d_idx in used_dets:
                        iou_matrix[t_idx, d_idx] = -1  # bỏ qua
                        continue

                    track_id = track_ids[t_idx]
                    det = detections[d_idx]

                    # Cập nhật track với detection mới
                    tr = self.tracks[track_id]
                    tr.bbox = det["bbox"]
                    tr.score = det["score"]
                    tr.class_name = det["class_name"]
                    tr.lost = 0

                    used_tracks.add(t_idx)
                    used_dets.add(d_idx)

                    iou_matrix[t_idx, d_idx] = -1

                # Detection chưa được dùng => tạo track mới
                for j, det in enumerate(detections):
                    if j not in used_dets:
                        self._create_track(det)

                # Track chưa được dùng => tăng lost
                for i, tid in enumerate(track_ids):
                    if i not in used_tracks:
                        tr = self.tracks.get(tid)
                        if tr is not None:
                            tr.lost += 1
            else:
                # Không có detection mới => tất cả track đều bị lost++
                for tr in self.tracks.values():
                    tr.lost += 1

        # Xóa những track đã mất quá lâu
        to_delete = [tid for tid, tr in self.tracks.items() if tr.lost > self.max_lost]
        for tid in to_delete:
            del self.tracks[tid]

        # Chuẩn hóa output
        outputs: List[Dict[str, Any]] = []
        for tid, tr in self.tracks.items():
            outputs.append(
                {
                    "id": tid,
                    "bbox": tr.bbox,
                    "score": tr.score,
                    "class_name": tr.class_name,
                    "lost": tr.lost,
                }
            )
        return outputs


