import os
import time
from typing import Callable, Dict, List, Any

import cv2
from utils.drawing import draw_detections_and_tracks


class FireEventManager:
    """
    Quản lý logic sự kiện cháy dựa trên kết quả tracking.

    Ý tưởng đơn giản:
    - Mỗi track ID được xem là một "object cháy" tiềm năng.
    - Nếu một track tồn tại liên tục >= min_frames frame (lost == 0 trong khoảng),
      ta coi đó là một sự kiện cháy thực sự và kích hoạt callback cảnh báo.
    - Sau khi một track đã kích hoạt sự kiện, ta không gửi lại quá thường xuyên
      nhờ cơ chế cooldown_seconds (tránh spam Telegram).
    - Ảnh lưu và gửi Telegram sẽ chứa TẤT CẢ các nhóm lửa được phát hiện tại thời điểm đó.
    """

    def __init__(
        self,
        min_frames: int,
        cooldown_seconds: int,
        on_event_confirmed: Callable[[str, Dict[str, Any]], None],
        save_dir: str = "outputs/events",
    ):
        """
        :param min_frames: số frame tối thiểu track phải xuất hiện để xác nhận cháy.
        :param cooldown_seconds: thời gian nghỉ giữa 2 lần gửi cảnh báo cho cùng một track.
        :param on_event_confirmed: callback khi sự kiện cháy được xác nhận.
                                  Hàm có dạng (image_path, meta_info_dict) -> None.
        :param save_dir: thư mục lưu ảnh minh họa sự kiện.
        """
        self.min_frames = min_frames
        self.cooldown_seconds = cooldown_seconds
        self.on_event_confirmed = on_event_confirmed
        self.save_dir = save_dir

        # Lưu trạng thái từng track
        self.track_frames_count: Dict[int, int] = {}  # track_id -> số frame xuất hiện
        self.track_last_alert_ts: Dict[int, float] = {}  # track_id -> timestamp lần gửi cảnh báo gần nhất

    def update(
        self, 
        detections: List[Dict[str, Any]],  # Thêm detections
        tracks: List[Dict[str, Any]], 
        frame_idx: int, 
        frame
    ) -> None:
        """
        Cập nhật trạng thái sự kiện dựa trên danh sách track ở frame hiện tại.

        :param detections: list[dict] chứa bbox, score, class_name - TẤT CẢ các nhóm lửa model phát hiện
        :param tracks: list[dict] chứa id, bbox, score, class_name, lost
        :param frame_idx: chỉ số frame hiện tại (phục vụ log/debug).
        :param frame: ảnh gốc để trích xuất và lưu làm minh họa.
        """
        now_ts = time.time()

        active_track_ids = set()
        confirmed_tracks = []  # Danh sách các track đã xác nhận sự kiện cháy trong frame này

        for tr in tracks:
            tid = tr["id"]
            lost = tr.get("lost", 0)
            active_track_ids.add(tid)

            if lost == 0:
                # Track đang nhìn thấy trong frame này
                self.track_frames_count[tid] = self.track_frames_count.get(tid, 0) + 1
            else:
                # Track tạm thời mất, nhưng chưa xóa; ta không tăng frame count
                pass

            # Kiểm tra xem track đã đủ điều kiện trở thành sự kiện cháy chưa
            count = self.track_frames_count.get(tid, 0)
            if count >= self.min_frames:
                # Kiểm tra cooldown để tránh spam cảnh báo cho cùng một track
                last_alert = self.track_last_alert_ts.get(tid, 0.0)
                if now_ts - last_alert >= self.cooldown_seconds:
                    # Thêm vào danh sách track đã xác nhận
                    confirmed_tracks.append(tr)
                    # Cập nhật thời gian gửi cảnh báo cuối cùng
                    self.track_last_alert_ts[tid] = now_ts

        # Nếu có ít nhất 1 track đã xác nhận sự kiện cháy, lưu ảnh và gửi cảnh báo
        if confirmed_tracks:
            # Lưu ảnh chứa TẤT CẢ các nhóm lửa được phát hiện
            image_path = self._save_event_image(
                frame, 
                frame_idx, 
                detections,  # Tất cả detections
                tracks,      # Tất cả tracks
                confirmed_tracks  # Các track đã xác nhận
            )

            # Tạo meta info với thông tin về tất cả các nhóm lửa
            meta = {
                "frame_idx": frame_idx,
                "num_detections": len(detections),  # Số lượng nhóm lửa model phát hiện
                "num_tracks": len(tracks),         # Số lượng tracks đang theo dõi
                "num_confirmed": len(confirmed_tracks),  # Số lượng track đã xác nhận sự kiện
                "confirmed_track_ids": [tr["id"] for tr in confirmed_tracks],  # Danh sách ID các track đã xác nhận
                "location": "Khu vực giám sát từ drone (demo)",
            }

            # Gọi callback gửi cảnh báo (VD: Telegram)
            self.on_event_confirmed(image_path, meta)

    def _save_event_image(
        self, 
        frame, 
        frame_idx: int, 
        detections: List[Dict[str, Any]],
        tracks: List[Dict[str, Any]],
        confirmed_tracks: List[Dict[str, Any]]
    ) -> str:
        """
        Lưu ảnh chứa TẤT CẢ các nhóm lửa được phát hiện tại thời điểm đó.
        
        Args:
            frame: Ảnh gốc
            frame_idx: Số frame
            detections: Tất cả detections từ YOLO
            tracks: Tất cả tracks đang theo dõi
            confirmed_tracks: Các track đã xác nhận sự kiện cháy
            
        Returns:
            Đường dẫn file ảnh đã lưu
        """
        os.makedirs(self.save_dir, exist_ok=True)

        # Vẽ tất cả detections và tracks lên frame
        vis_frame = draw_detections_and_tracks(
            frame,
            detections=detections,  # Vẽ tất cả detections (màu cyan)
            tracks=tracks,          # Vẽ tất cả tracks (màu xanh lá)
            show_ids=True,
            show_detections=True,
            show_tracks=True,
        )
        
        # Vẽ thêm thông tin cảnh báo ở góc trên bên trái
        warning_text = f"FIRE ALERT - Frame {frame_idx}"
        warning_text2 = f"Detections: {len(detections)}, Tracks: {len(tracks)}, Confirmed: {len(confirmed_tracks)}"
        
        # Vẽ background cho text cảnh báo
        (text_width1, text_height1), _ = cv2.getTextSize(
            warning_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
        )
        (text_width2, text_height2), _ = cv2.getTextSize(
            warning_text2, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        
        max_width = max(text_width1, text_width2)
        total_height = text_height1 + text_height2 + 20
        
        cv2.rectangle(
            vis_frame,
            (10, 10),
            (10 + max_width + 20, 10 + total_height),
            (0, 0, 255),  # Màu đỏ
            -1,
        )
        
        # Vẽ text cảnh báo
        cv2.putText(
            vis_frame,
            warning_text,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),  # Màu trắng
            2,
            cv2.LINE_AA,
        )
        
        cv2.putText(
            vis_frame,
            warning_text2,
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),  # Màu trắng
            2,
            cv2.LINE_AA,
        )

        # Tên file với timestamp
        filename = f"fire_alert_frame{frame_idx}_{int(time.time())}.jpg"
        save_path = os.path.join(self.save_dir, filename)
        cv2.imwrite(save_path, vis_frame)
        return save_path


