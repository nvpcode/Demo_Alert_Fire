import cv2


class VideoStream:
    """
    Lớp tiện ích để đọc khung hình từ video / webcam.

    - source: có thể là int (webcam) hoặc đường dẫn file video / URL RTSP.
    - iterator trả về (frame_idx, frame) để dễ debug và quản lý.
    """

    def __init__(self, source=0):
        self.source = source
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Không mở được nguồn video: {source}")
        self.frame_idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        ok, frame = self.cap.read()
        if not ok:
            raise StopIteration
        idx = self.frame_idx
        self.frame_idx += 1
        return idx, frame

    def release(self):
        """
        Giải phóng tài nguyên video.
        """
        if self.cap is not None:
            self.cap.release()


