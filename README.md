## Hệ thống phát hiện & cảnh báo cháy từ video drone

Dự án này là một MVP end-to-end dùng **YOLO11n** để phát hiện ngọn lửa trên video quay từ drone và gửi cảnh báo qua **Telegram**.

### Thành phần chính
- **YOLO11n**: phát hiện vùng cháy trên từng khung hình.
- **Tracker (IoU / ByteTrack)**: theo dõi ID của vùng cháy qua nhiều khung hình.
- **Fire Event Manager**: gộp các track bền vững thành một sự kiện cháy.
- **Telegram Notifier**: gửi ảnh khung hình + thông tin khu vực quay lên Telegram.

### Cấu trúc thư mục (src/)
- `main.py`: điểm chạy chính, nối toàn bộ pipeline.
- `configs/config.yaml`: cấu hình chung (đường dẫn model, ngưỡng, Telegram, video input...).
- `detection/yolo_detector.py`: lớp wrapper cho YOLOv8.
- `tracking/iou_tracker.py`: tracker đơn giản dựa trên IoU.
- `events/fire_event_manager.py`: quản lý logic sự kiện cháy.
- `notifiers/telegram_notifier.py`: gửi cảnh báo qua Telegram Bot.
- `video/video_stream.py`: đọc video từ file / webcam / RTSP.
- `utils/drawing.py`: vẽ bounding box, ID, trạng thái lên frame.
- `utils/logging_utils.py`: hàm log đơn giản.

### Cách cài đặt
```bash
pip install -r requirements.txt
```

### Cách chạy (ví dụ)
```bash
python src/main.py --config configs/config.yaml
```

> Lưu ý: Bạn cần tự thiết lập **TELEGRAM_BOT_TOKEN** và **TELEGRAM_CHAT_ID** trong `configs/config.yaml`.


