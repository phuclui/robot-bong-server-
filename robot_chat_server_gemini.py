"""
Server mẫu gọi Gemini API (miễn phí) cho robot trò chuyện.
Cài thư viện trước:  pip install google-genai flask

Lấy API key miễn phí tại: https://aistudio.google.com/app/apikey
(không cần thẻ tín dụng)

Chạy:  python robot_chat_server.py
"""

import os
import asyncio
import edge_tts
from google import genai
from google.genai import types
from flask import Flask, request, jsonify, Response

# ----- CẤU HÌNH -----
# Đặt API key qua biến môi trường, KHÔNG hardcode trong code:
#   export GEMINI_API_KEY="AIzaSy..."   (Linux/Mac)
#   set GEMINI_API_KEY="AIzaSy..."      (Windows)
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

MODEL = "gemini-flash-lite-latest"   # quota free tier cao hơn hẳn bản Flash thường

# ----- TÍNH CÁCH ROBOT -----
SYSTEM_PROMPT = """\
Bạn là trợ lý robot tên Bống, sống trong một con robot đồ chơi biết nói.
Tính cách: vui vẻ, dí dỏm, hơi đanh đá kiểu em gái lanh chanh, hay trêu chọc
nhẹ nhàng nhưng không bao giờ ác ý hay xúc phạm người dùng.
Quy tắc:
- Trả lời ngắn gọn (1-3 câu), vì đây là hội thoại nói, không phải văn bản dài.
- Xưng "em", gọi người dùng là "anh/chị" hoặc theo cách họ xưng hô.
- Có thể chêm chút hài hước, mỉa mai nhẹ, nhưng luôn giữ thiện chí.
- Không dùng markdown, emoji, hay ký hiệu đặc biệt vì output sẽ được đọc thành giọng nói.
"""

GEN_CONFIG = types.GenerateContentConfig(
    system_instruction=SYSTEM_PROMPT,
    max_output_tokens=300,
    temperature=0.9,   # cao hơn chút để câu trả lời có màu sắc, ít khô khan
)

# ----- LƯU PHIÊN CHAT THEO TỪNG ROBOT/SESSION -----
# Đối tượng `chat` của Gemini tự động giữ lịch sử hội thoại cho mỗi phiên.
# Với server thật, nên dùng database/Redis thay vì dict trong RAM.
chat_sessions = {}


def get_or_create_chat(session_id: str):
    if session_id not in chat_sessions:
        chat_sessions[session_id] = client.chats.create(
            model=MODEL,
            config=GEN_CONFIG,
        )
    return chat_sessions[session_id]


def chat_with_gemini(session_id: str, user_message: str) -> str:
    chat = get_or_create_chat(session_id)
    response = chat.send_message(user_message)
    return response.text


# ----- NHẬN DẠNG GIỌNG NÓI (SPEECH-TO-TEXT) -----
# Gemini hỗ trợ nhận input là audio trực tiếp, nên không cần thêm dịch vụ
# STT riêng - dùng chung 1 API key, vẫn nằm trong free tier.
TRANSCRIBE_PROMPT = (
    "Hãy chuyển đoạn âm thanh sau thành văn bản tiếng Việt chính xác. "
    "Chỉ trả về đúng nội dung lời nói, không thêm giải thích, không thêm "
    "dấu ngoặc kép, không thêm bất kỳ chữ nào khác."
)


def transcribe_audio(wav_bytes: bytes) -> str:
    response = client.models.generate_content(
        model=MODEL,
        contents=[
            TRANSCRIBE_PROMPT,
            types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav"),
        ],
    )
    return response.text.strip()


# ----- CHUYỂN VĂN BẢN THÀNH GIỌNG NÓI (TEXT-TO-SPEECH) -----
# Dùng edge-tts (giọng đọc của Microsoft Edge) - HOÀN TOÀN MIỄN PHÍ,
# không giới hạn số lượt/ngày, không cần API key riêng.
# Xem danh sách giọng: chạy lệnh `edge-tts --list-voices` hoặc xem
# https://github.com/rany2/edge-tts
TTS_VOICE = "vi-VN-HoaiMyNeural"   # giọng nữ miền Bắc, tự nhiên
# TTS_VOICE = "vi-VN-NamMinhNeural"  # giọng nam, nếu muốn đổi


def synthesize_speech_mp3(text: str) -> bytes:
    """Chuyển text thành audio, trả về bytes dạng MP3."""
    async def _generate():
        communicate = edge_tts.Communicate(text, voice=TTS_VOICE)
        audio_bytes = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_bytes += chunk["data"]
        return audio_bytes

    return asyncio.run(_generate())


# ----- API ENDPOINT CHO ROBOT GỌI -----
app = Flask(__name__)


@app.route("/chat", methods=["POST"])
def chat_endpoint():
    data = request.get_json(force=True)
    session_id = data.get("session_id", "default")
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"error": "Thiếu 'message'"}), 400

    try:
        reply = chat_with_gemini(session_id, user_message)
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/transcribe", methods=["POST"])
def transcribe_endpoint():
    # ESP32 gửi file WAV thô trong body, header Content-Type: audio/wav
    wav_bytes = request.get_data()

    if not wav_bytes:
        return "Thiếu dữ liệu âm thanh", 400

    try:
        text = transcribe_audio(wav_bytes)
        # Trả về text thuần (không phải JSON) để code ESP32 đọc trực tiếp
        # bằng http.getString() như đã viết ở file .ino
        return text, 200, {"Content-Type": "text/plain; charset=utf-8"}
    except Exception as e:
        return f"Lỗi transcribe: {e}", 500


@app.route("/tts", methods=["GET", "POST"])
def tts_endpoint():
    # Hỗ trợ cả GET (?text=...) để ESP32 có thể mở thẳng URL này bằng
    # thư viện giải mã MP3, và POST (JSON) cho các client khác.
    if request.method == "GET":
        text = request.args.get("text", "")
    else:
        data = request.get_json(force=True)
        text = data.get("text", "")

    if not text:
        return jsonify({"error": "Thiếu 'text'"}), 400

    try:
        mp3_bytes = synthesize_speech_mp3(text)
        # Trả về file MP3 thô, ESP32 dùng thư viện ESP32-audioI2S để giải mã trực tiếp
        return Response(mp3_bytes, mimetype="audio/mpeg")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # host="0.0.0.0" để robot trong cùng mạng LAN có thể gọi tới server này
    app.run(host="0.0.0.0", port=5000, debug=True)
