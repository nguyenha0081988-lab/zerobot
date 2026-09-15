import os
import re
import json
import asyncio
import urllib.parse
from datetime import datetime
from typing import List, Dict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import gradio as gr
from groq import Groq
import httpx

# 1. Khởi tạo FastAPI app
app = FastAPI()

# 2. Khởi tạo Groq Client
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

def get_groq_client():
    key = os.getenv("GROQ_API_KEY", GROQ_API_KEY)
    if not key:
        return None
    return Groq(api_key=key)

def detect_available_model():
    client = get_groq_client()
    if not client:
        return "llama-3.1-8b-instant"
    try:
        models = client.models.list()
        model_ids = [m.id for m in models.data]
        preferred_models = [
            "llama-3.1-8b-instant",
            "openai/gpt-oss-20b",
            "qwen/qwen3.8-27b"
        ]
        for pref in preferred_models:
            if pref in model_ids:
                return pref
        if model_ids:
            return model_ids[0]
    except Exception as e:
        print(f"Error detecting model: {e}")
    return "llama-3.1-8b-instant"

# 3. Phân loại ý định: Người dùng có đang hỏi thời tiết không?
def is_weather_query(prompt: str) -> bool:
    keywords = [
        "thời tiết", "nhiệt độ", "bao nhiêu độ", "mưa", "nắng", 
        "dự báo", "khí hậu", "thời tiết ở", "thời tiết tại", "có mưa không",
        "mặc gì", "mang ô", "mang áo mưa"
    ]
    prompt_lower = prompt.lower()
    return any(kw in prompt_lower for kw in keywords)

# 4. Trích xuất địa điểm & Gọi API Thời tiết
def extract_location_query(prompt: str) -> str:
    text = prompt.strip()
    patterns_to_remove = [
        r"^(cho hỏi|hỏi|xem|kiểm tra|dự báo|thời tiết|thời tiết ở|thời tiết tại|ở|tại|khu vực|thành phố|tỉnh)\s+",
        r"\s+(thì sao|thế nào|như thế nào|ra sao|có mưa không|mưa không|bao nhiêu độ|thế|à|hả|với)$"
    ]
    for pattern in patterns_to_remove:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
    return text if text else "Phúc Yên"

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
                return f"Thời tiết thực tế tại {location_name}: Nhiệt độ {temp}°C, độ ẩm {humidity}%, {rain_status}."
    except Exception as e:
        print(f"Weather API error: {e}")

    return f"Thời tiết tại {location_name}: Nhiệt độ khoảng 27°C, trời có mây."

# 5. Core AI Processor với Quản lý Ngữ cảnh & System Prompt chuẩn Người thật
async def process_chat(messages_history: List[Dict[str, str]], current_prompt: str) -> str:
    client = get_groq_client()
    if not client:
        return "Lỗi: Chưa cấu hình GROQ_API_KEY trên Render Environment Variables."

    now = datetime.now()
    days_vn = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]
    current_day = days_vn[now.weekday()]
    current_time_str = f"{current_day}, ngày {now.strftime('%d/%m/%Y')}, lúc {now.strftime('%H:%M:%S')}"
    
    # Chỉ gọi API thời tiết khi người dùng thực sự hỏi/liên quan
    weather_context = ""
    if is_weather_query(current_prompt):
        weather_info = await get_dynamic_weather(current_prompt)
        weather_context = f"\n[DỮ LIỆU THỜI TIẾT THỰC TẾ TRONG HỆ THỐNG]:\n{weather_info}\n"

    system_instruction = f"""Bạn là VietBot - một trợ lý AI thông minh, tinh tế và cực kỳ tự nhiên.
Thời gian thực tế: {current_time_str}.{weather_context}

NGUYÊN TẮC GIAO TIẾP TỰ NHIÊN (HUMAN-LIKE PERSONA):
1. TỰ NHIÊN VÀ ẤM ÁP: Trò chuyện chân thành, dùng ngôn ngữ tiếng Việt đời thường tự nhiên. Thêm từ đệm hợp lý khi giao tiếp (như "Dạ", "À", "Ồ", "Vâng", "Đúng rồi bạn", "Dạ được chứ").
2. TUYỆT ĐỐI KHÔNG MÁY MÓC: KHÔNG xưng "Tôi là một mô hình AI/LLM", không trả lời theo kiểu lập trình rập khuôn. Khi được chào hỏi ("xin chào", "hi"), hãy đáp lại thân thiện và hỏi xem có thể hỗ trợ gì, KHÔNG tự ý báo cáo thời tiết hay dữ liệu hệ thống khi không được hỏi.
3. ĐỒNG CẢM & THẤU HỂU: Nhận biết cảm xúc trong lời nói của người dùng để phản hồi tinh tế trước khi giải quyết vấn đề.
4. TẬP TRUNG NGẮN GỌN: Khi trò chuyện thông thường, hãy trả lời cô đọng, vừa đủ như hai người bạn đang nhắn tin với nhau. Chỉ trình bày chi tiết/có cấu trúc khi được yêu cầu phân tích, viết code hoặc hướng dẫn công việc.
5. NHỚ NGỮ CẢNH: Theo dõi sát luồng hội thoại phía trước để phản hồi nhất quán."""

    selected_model = detect_available_model()

    # Xây dựng danh sách messages bao gồm history (tối đa 10 lượt gần nhất)
    full_messages = [{"role": "system", "content": system_instruction}]
    
    # Lấy tối đa 10 tin nhắn gần nhất từ lịch sử
    recent_history = messages_history[-10:] if len(messages_history) > 10 else messages_history
    for msg in recent_history:
        full_messages.append({"role": msg["role"], "content": msg["content"]})
        
    full_messages.append({"role": "user", "content": current_prompt})

    def _call_groq():
        completion = client.chat.completions.create(
            model=selected_model,
            messages=full_messages,
            temperature=0.5,
            max_tokens=600
        )
        return completion.choices[0].message.content

    try:
        return await asyncio.to_thread(_call_groq)
    except Exception as e:
        return f"Rất tiếc, hệ thống đang gặp chút sự cố kết nối: {str(e)}"

# 6. Giao diện Gradio Chat với bộ nhớ hội thoại
async def chat_fn(message, history):
    # Convert history từ định dạng Gradio sang format API
    formatted_history = []
    for user_msg, bot_msg in history:
        if user_msg:
            formatted_history.append({"role": "user", "content": user_msg})
        if bot_msg:
            formatted_history.append({"role": "assistant", "content": bot_msg})
            
    return await process_chat(formatted_history, message)

demo = gr.ChatInterface(
    fn=chat_fn,
    title="🤖 VietBot AI Platform",
    description="Trợ lý AI Giao tiếp Thông minh & Tự nhiên (Hỗ trợ Web, App Mobile & ESP32)."
)

# 7. WebSocket Endpoint có Quản lý Session History riêng cho từng Client
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    client_history: List[Dict[str, str]] = []
    
    try:
        while True:
            data = await websocket.receive_text()
            user_text = data
            try:
                json_data = json.loads(data)
                user_text = json_data.get("text") or json_data.get("message") or json_data.get("content") or data
            except Exception:
                pass
            
            clean_text = str(user_text).strip()
            if not clean_text:
                continue
                
            ai_reply = await process_chat(client_history, clean_text)
            
            # Cập nhật bộ nhớ hội thoại của client này
            client_history.append({"role": "user", "content": clean_text})
            client_history.append({"role": "assistant", "content": ai_reply})
            
            # Giới hạn lịch sử lưu tối đa 20 tin nhắn trong bộ nhớ tạm
            if len(client_history) > 20:
                client_history = client_history[-20:]
                
            await websocket.send_text(ai_reply)
    except WebSocketDisconnect:
        pass

# Gắn Gradio app vào FastAPI root
app = gr.mount_gradio_app(app, demo, path="/")
