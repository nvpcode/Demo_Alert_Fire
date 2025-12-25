import argparse
import os

import cv2
import yaml
from ultralytics import YOLO

from events.fire_event_manager import FireEventManager
from notifiers.telegram_notifier import TelegramNotifier
from utils.drawing import draw_detections_and_tracks
from utils.logging_utils import get_logger


# ==========================
# Điểm vào chính của hệ thống
# ==========================

def load_config(config_path: str) -> dict:
    """
    Đọc file cấu hình YAML.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args():
    """
    Parse tham số dòng lệnh.
    """
    parser = argparse.ArgumentParser(description="Hệ thống phát hiện & theo dõi cháy từ video drone")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Đường dẫn file cấu hình YAML",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    logger = get_logger("fire_drone_system")

    # ==== Khởi tạo các thành phần từ config ====
    video_cfg = config.get("video", {})
    yolo_cfg = config.get("yolo", {})
    tracker_cfg = config.get("tracker", {})
    event_cfg = config.get("event", {})
    telegram_cfg = config.get("telegram", {})

    # Video input
    video_source = video_cfg.get("source", 0)  # 0: webcam, hoặc đường dẫn file
    window_name = video_cfg.get("window_name", "Fire Drone Monitoring")
    show_window = video_cfg.get("show_window", True)
    save_processed = video_cfg.get("save_processed", True)
    output_video_path = video_cfg.get("output_path", "outputs/video_processed/output.mp4")
    output_fps = video_cfg.get("output_fps", None)  # nếu None sẽ lấy từ nguồn hoặc mặc định 25

    # Khởi tạo YOLO (dùng track tích hợp ByteTrack)
    model = YOLO(yolo_cfg.get("weights", "yolov8n.pt"))
    logger.info(f"Đã tải model YOLOv8 từ: {yolo_cfg.get('weights', 'yolov8n.pt')}")

    # Notifier (Telegram)
    notifier = TelegramNotifier(
        bot_token=telegram_cfg.get("bot_token", ""),
        chat_id=telegram_cfg.get("chat_id", ""),
        enabled=telegram_cfg.get("enabled", False),
    )

    # Fire Event Manager
    event_manager = FireEventManager(
        min_frames=event_cfg.get("min_frames", 10),
        cooldown_seconds=event_cfg.get("cooldown_seconds", 60),
        on_event_confirmed=notifier.send_fire_alert,
        save_dir=event_cfg.get("save_dir", "outputs/events"),
    )

    os.makedirs(event_manager.save_dir, exist_ok=True)

    logger.info("Bắt đầu xử lý video...")

    # Chuẩn bị VideoWriter nếu cần lưu video đã xử lý
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    writer = None

    # ==== Vòng lặp xử lý từng khung hình bằng YOLO.track (ByteTrack tích hợp) ====
    results = model.track(
        source=video_source,
        stream=True,
        tracker=tracker_cfg.get("config", "src/tracking/bytetrack.yaml"),
        conf=yolo_cfg.get("conf_thres", 0.3),
        iou=yolo_cfg.get("iou_thres", 0.5),
        device=yolo_cfg.get("device", "cpu"),
        show=False,
    )

    for frame_idx, r in enumerate(results):
        frame = r.orig_img
        if frame is None:
            logger.warning("Không lấy được frame, dừng lại.")
            break

        boxes = r.boxes
        detections = []
        tracks = []

        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy()
            ids = boxes.id.cpu().numpy() if boxes.id is not None else None
            names = r.names

            for i in range(len(xyxy)):
                class_name = names.get(int(cls_ids[i]), "fire") if hasattr(names, "get") else "fire"
                det_dict = {
                    "bbox": xyxy[i].tolist(),
                    "score": float(confs[i]),
                    "class_name": class_name,
                }
                detections.append(det_dict)

                if ids is not None:
                    tid = int(ids[i])
                    tracks.append(
                        {
                            "id": tid,
                            "bbox": xyxy[i].tolist(),
                            "score": float(confs[i]),
                            "class_name": class_name,
                            "lost": 0,
                            "state": "tracked",
                        }
                    )

        # 3. Quản lý sự kiện cháy - TRUYỀN CẢ DETECTIONS VÀ TRACKS
        event_manager.update(detections, tracks, frame_idx, frame)

        # 4. Vẽ thông tin lên frame - VẼ CẢ DETECTIONS VÀ TRACKS
        vis_frame = draw_detections_and_tracks(
            frame,
            detections=detections,
            tracks=tracks,
            show_ids=True,
            show_detections=True,
            show_tracks=True,
        )

        # 5. Lưu video đã xử lý nếu bật
        if save_processed:
            if writer is None:
                h, w = vis_frame.shape[:2]
                fps = output_fps
                if fps is None:
                    try:
                        cap = cv2.VideoCapture(video_source)
                        fps_val = cap.get(cv2.CAP_PROP_FPS)
                        cap.release()
                        fps = fps_val if fps_val and fps_val > 1e-3 else 25.0
                    except Exception:
                        fps = 25.0
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
            writer.write(vis_frame)

        # 6. Hiển thị nếu bật
        if show_window:
            cv2.imshow(window_name, vis_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC để thoát
                logger.info("Nhấn ESC, dừng xử lý.")
                break

    try:
        cv2.destroyAllWindows()
    except Exception:
        # Trong trường hợp không dùng được GUI (server headless), bỏ qua
        pass

    if writer is not None:
        writer.release()

    logger.info("Kết thúc chương trình.")


if __name__ == "__main__":
    main()


