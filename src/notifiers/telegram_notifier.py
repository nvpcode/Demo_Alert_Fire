import asyncio
import logging
import threading
from typing import Dict, Any

from telegram import Bot
from telegram.constants import ParseMode


class TelegramNotifier:
    """
    Gửi cảnh báo cháy qua Telegram Bot.

    - Nếu không cấu hình token / chat_id hoặc disabled, class này sẽ log cảnh báo
      thay vì ném exception (để hệ thống vẫn chạy được).
    - Sử dụng event loop riêng trong background thread để tránh lỗi "Event loop is closed".
    """

    def __init__(self, bot_token: str, chat_id: str, enabled: bool = True):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = enabled and bool(bot_token) and bool(chat_id)

        self.logger = logging.getLogger("TelegramNotifier")

        self.bot = None
        self.loop = None
        self.loop_thread = None
        
        if self.enabled:
            try:
                self.bot = Bot(token=self.bot_token)
                # Tạo event loop riêng trong background thread
                self._start_event_loop()
            except Exception as e:
                self.logger.error(f"Không thể khởi tạo Telegram Bot: {e}")
                self.enabled = False

    def _start_event_loop(self):
        """
        Tạo event loop riêng trong background thread để tránh lỗi "Event loop is closed".
        """
        def run_event_loop():
            """Chạy event loop trong thread riêng"""
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_forever()
        
        self.loop_thread = threading.Thread(target=run_event_loop, daemon=True)
        self.loop_thread.start()
        
        # Đợi một chút để event loop khởi động
        import time
        time.sleep(0.1)

    def send_fire_alert(self, image_path: str, meta: Dict[str, Any]) -> None:
        """
        Hàm callback được gọi khi sự kiện cháy được xác nhận.

        :param image_path: đường dẫn ảnh minh họa (frame chứa TẤT CẢ các nhóm lửa).
        :param meta: dict chứa thông tin bổ sung (số lượng detections, tracks, v.v.).
        """
        if not self.enabled or self.bot is None:
            self.logger.warning(
                f"TelegramNotifier chưa được bật hoặc thiếu cấu hình. "
                f"Giả lập gửi cảnh báo: image={image_path}, meta={meta}"
            )
            return

        # Tạo caption với thông tin về TẤT CẢ các nhóm lửa
        caption_lines = [
            "🔥 *CẢNH BÁO CHÁY PHÁT HIỆN TỪ VIDEO DRONE* 🔥",
            "",
            f"📊 *Thống kê:*",
            f"- Số nhóm lửa phát hiện: `{meta.get('num_detections', 0)}`",
            f"- Số track đang theo dõi: `{meta.get('num_tracks', 0)}`",
            f"- Số track đã xác nhận: `{meta.get('num_confirmed', 0)}`",
            "",
            f"🆔 *Track IDs đã xác nhận:* `{', '.join(map(str, meta.get('confirmed_track_ids', [])))}`",
            "",
            f"📹 Frame: `{meta.get('frame_idx', 'N/A')}`",
            f"📍 Khu vực: {meta.get('location', 'Không rõ')}",
        ]
        caption = "\n".join(caption_lines)

        try:
            # Sử dụng event loop riêng để chạy coroutine
            if self.loop is None or self.loop.is_closed():
                self.logger.error("Event loop không khả dụng, không thể gửi Telegram")
                return
            
            # Lên lịch coroutine trong event loop riêng
            future = asyncio.run_coroutine_threadsafe(
                self._send_photo_async(image_path, caption),
                self.loop
            )
            # Đợi kết quả (có thể set timeout)
            future.result(timeout=10)  # Timeout 10 giây
            self.logger.info(f"Đã gửi cảnh báo cháy lên Telegram: {image_path}")
        except Exception as e:
            self.logger.error(f"Lỗi khi gửi ảnh cảnh báo lên Telegram: {e}")

    async def _send_photo_async(self, image_path: str, caption: str) -> None:
        """
        Hàm async helper để gửi ảnh lên Telegram.
        """
        try:
            with open(image_path, "rb") as f:
                await self.bot.send_photo(
                    chat_id=self.chat_id,
                    photo=f,
                    caption=caption,
                    parse_mode=ParseMode.MARKDOWN,
                )
        except Exception as e:
            self.logger.error(f"Lỗi trong _send_photo_async: {e}")
            raise

    def __del__(self):
        """
        Cleanup: đóng event loop khi object bị hủy.
        """
        if self.loop is not None and not self.loop.is_closed():
            self.loop.call_soon_threadsafe(self.loop.stop)


