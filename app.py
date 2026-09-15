import os
import re
import json
import asyncio
import urllib.parse
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect
import gradio as gr
from groq import Groq
import httpx

# 1. Khởi tạo Groq Client (Lấy key an toàn từ biến môi trường)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

def detect_available_model():
    try:
        models = client.models.list()
        model_ids = [m.id for m in models.data]
        preferred_models = [
            "openai/gpt-oss-20b",
            "openai/gpt-oss-120b",
            "qwen/qwen3.8-27b",
            "llama-3.1-8b-instant"
        ]
        for pref in preferred_models:
            if pref in model_ids:
                return pref
        if model_ids:
            return model_ids[0]
    except Exception:
        pass
    return "openai/gpt-oss-20b"

SELECTED_MODEL = detect_available_model()

# 2. Trích xuất địa điểm & Gọi API Thời tiết
def extract_location_query(prompt: str) -> str:
    text = prompt.strip()
    patterns_to_remove = [
        r"^(cho hỏi|hỏi|xem|kiểm tra|dự báo|thời tiết|thời tiết ở|thời tiết tại|ở|tại|khu vực|thành phố|tỉnh)\s+",
        r"\s+(thì sao|thế nào|như thế nào|ra sao|có mưa không|mưa không|bao nhiêu độ|thế|à|hả|với)$"
    ]
    for pattern in patterns_to_remove:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
    return text if text else prompt

async def get_dynamic_weather(prompt: str) -> str:
    location_name = "Phúc Yên, Vĩnh Phúc"
    lat, lon = 21.3188, 105.8072
    extracted_loc = extract_location_query(prompt)

    if any(k in prompt.lower() for k in ["chỗ tôi", "ở đây", "tại đây", "hiện tại", "chỗ mình"]):
        extracted_loc = "Phúc Yên"

    try:
        encoded_loc = urllib.parse.quote(extracted_loc)
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={encoded_loc}&count=1&language=vi&format=json"
        
        async with httpx.AsyncClient(timeout=3.0) as http_client:
            geo_resp = await http_client.get(geo_url)
            if geo_resp.status_code == 200:
                geo_data = geo_resp.json().get("results")
                if geo_data:
                    target = geo_data[0]
                    lat = target.get("latitude")
                    lon = target.get("longitude")
                    country = target.get("country", "")
                    admin1 = target.get("admin1", "")
                    name = target.get("name", extracted_loc)
                    loc_parts = [p for p in [name, admin1, country] if p]
                    location_name = ", ".join(loc_parts)
    except Exception as e:
        print(f"Geocoding error: {e}")

    try:
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation,weather_code&timezone=auto"
        async with httpx.AsyncClient(timeout=3.0) as http_client:
            resp = await http_client.get(weather_url)
            if resp.status_code == 200:
                data = resp.json().get("current", {})
                temp = data.get("temperature_2m", 27)
                precip = data.get("precipitation", 0)
                humidity = data.get("relative_humidity_2m", 80)
                rain_status = "đang có mưa" if precip > 0 else "không mưa, nhiều mây/có mây"
                return f"Thời tiết thực tế tại {location_name}: Nhiệt độ {temp}°C, độ ẩm {humidity}%, trạng thái: {rain_status}."
    except Exception as e:
        print(f"Weather API error: {e}")

    return f"Thời tiết tại {location_name}: Nhiệt độ khoảng 27°C, trời có mây."

# 3. Trí tuệ nhân tạo Groq LLM
async def get_groq_response(prompt: str) -> str:
    if not prompt or not prompt.strip():
        return "Xin chào! Tôi có thể giúp gì cho bạn?"

    now = datetime.now()
    days_vn = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]
    current_day = days_vn[now.weekday()]
    current_time_str = f"{current_day}, ngày {now.strftime('%d/%m/%Y')}, lúc {now.strftime('%H:%M:%S')}"
    
    weather_info = await get_dynamic_weather(prompt)

    system_instruction = f"""Bạn là VietBot AI - Trợ lý thông minh bằng tiếng Việt.
Thời gian hiện tại: {current_time_str}.

[DỮ LIỆU THỜI TIẾT THỰC TẾ TRUY VẤN ĐƯỢC]:
{weather_info}

Nhiệm vụ:
1. Bạn ĐÃ CÓ sẵn dữ liệu thời tiết thực tế ở trên. Hãy dùng dữ liệu này để trả lời người dùng.
2. Tuyệt đối KHÔNG ĐƯỢC trả lời "không có dữ liệu" hoặc "chỉ có dữ liệu Phúc Yên".
3. Trả lời ngắn gọn, tự nhiên, chính xác theo đúng dữ liệu được cung cấp."""

    def _call_groq():
        completion = client.chat.completions.create(
            model=SELECTED_MODEL,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=512
        )
        return completion.choices[0].message.content

    return await asyncio.to_thread(_call_groq)

# 4. Giao diện Chat Gradio
async def chat_fn(message, history):
    return await get_groq_response(message)

demo = gr.ChatInterface(
    fn=chat_fn,
    title="🤖 VietBot AI Platform",
    description="Server AI hỗ trợ xử lý giọng nói & nhắn tin cho Mobile App."
)

# 5. Đăng ký FastAPI app & tuyến WebSocket cho Uvicorn
app = demo.app

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            user_text = data
            try:
                json_data = json.loads(data)
                user_text = json_data.get("text") or json_data.get("message") or json_data.get("content") or data
            except Exception:
                pass
            
            if not str(user_text).strip():
                continue
                
            ai_reply = await get_groq_response(str(user_text))
            await websocket.send_text(ai_reply)
    except WebSocketDisconnect:
        pass
