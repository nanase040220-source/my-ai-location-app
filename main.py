import asyncio
import os
import json
import base64
import re
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from google import genai
from google.genai import types
import httpx

app = FastAPI()
templates = Jinja2Templates(directory="templates")

gemini_key = os.environ.get("GEMINI_API_KEY")
groq_key = os.environ.get("GROQ_API_KEY")

try:
    if gemini_key:
        gemini_client = genai.Client(api_key=gemini_key)
    else:
        gemini_client = None
except Exception as e:
    print(f"Gemini Init Error: {e}")
    gemini_client = None

def optimize_image(image_bytes: bytes) -> bytes:
    """🛠️ Groqの400エラー(容量オーバー)を防ぐための画像自動圧縮"""
    try:
        img = Image.open(BytesIO(image_bytes))
        # 透過PNGなどでエラーが出ないようRGBに変換
        if img.mode != "RGB":
            img = img.convert("RGB")
        # Groqが安全に読み込める800pxに縮小
        img.thumbnail((800, 800))
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        return buffer.getvalue()
    except Exception as e:
        print(f"画像圧縮エラー: {e}")
        return image_bytes

async def ask_gemini_geoguessr(image_bytes: bytes, mime_type: str):
    """メインAI: Google Gemini"""
    if not gemini_client:
        raise Exception("Gemini API Key is missing.")

    prompt = (
        "画像から場所を特定してください。必ず以下のJSONフォーマットのみで返してください。\n"
        "{\n"
        "  \"reason\": \"推論の理由（日本語）\",\n"
        "  \"query_used\": \"検索に使ったキーワード\",\n"
        "  \"location\": \"特定された住所や地名\",\n"
        "  \"lat\": 緯度(float),\n"
        "  \"lng\": 経度(float)\n"
        "}"
    )

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=0.2
    )
    
    response = await gemini_client.aio.models.generate_content(
        model='gemini-2.0-flash',
        contents=[types.Part.from_bytes(data=image_bytes, mime_type=mime_type), prompt],
        config=config
    )
    return response.text

async def ask_groq_geoguessr(image_bytes: bytes):
    """バックアップAI: Groq (400エラー対策済)"""
    if not groq_key:
        raise Exception("Groq API Key is missing.")

    prompt = (
        "画像から場所を特定してください。以下のタグを使ってテキストで回答してください。\n"
        "<reason>推論の理由</reason>\n"
        "<query_used>検索キーワード</query_used>\n"
        "<location>住所や地名</location>\n"
        "<lat>緯度</lat>\n"
        "<lng>経度</lng>"
    )

    # 🚀 ここで画像を圧縮してGroqのサイズ制限をクリアする
    optimized_bytes = optimize_image(image_bytes)
    base64_image = base64.b64encode(optimized_bytes).decode('utf-8')
    data_url = f"data:image/jpeg;base64,{base64_image}"

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"}
    payload = {
        "model": "llama-3.2-11b-vision-preview",
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": data_url}}]}],
        "temperature": 0.2,
        "max_tokens": 1024 # 🚀 400エラー対策：トークン上限を明示
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=payload, timeout=30.0)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

def parse_groq_text(text: str):
    """Groqのテキストをフロント画面向けに変換"""
    def extract(tag, default=""):
        match = re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
        return match.group(1).strip() if match else default

    try:
        return {
            "reason": extract("reason", "詳細不明") + " (※バックアップAIで解析)",
            "query_used": extract("query_used", "画像検索"),
            "location": extract("location", "推定エリア"),
            "lat": float(extract("lat", "35.6812")),
            "lng": float(extract("lng", "139.7671"))
        }
    except:
        return {"reason": "解析エラー", "query_used": "-", "location": "不明", "lat": 35.6, "lng": 139.7}

@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def read_index(request: Request):
    if request.method == "HEAD":
        return HTMLResponse(content="", status_code=200)
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    mime_type = file.content_type or "image/jpeg"

    try:
        # 1. まずはメインのGeminiへ
        gemini_result = await ask_gemini_geoguessr(image_bytes, mime_type)
        data = json.loads(gemini_result)
        return {
            "success": True,
            "reason": data.get("reason", ""),
            "query_used": data.get("query_used", ""),
            "location": data.get("location", ""),
            "lat": float(data.get("lat", 0)),
            "lng": float(data.get("lng", 0))
        }
    except Exception as e:
        print(f"Gemini混雑、Groqへ移行: {e}")
        if groq_key:
            try:
                # 2. Geminiがダメなら、自動圧縮した上でGroqへ
                groq_text = await ask_groq_geoguessr(image_bytes)
                data = parse_groq_text(groq_text)
                return {
                    "success": True,
                    "reason": data["reason"],
                    "query_used": data["query_used"],
                    "location": data["location"],
                    "lat": data["lat"],
                    "lng": data["lng"]
                }
            except Exception as e2:
                return {"success": False, "message": f"すべてのAIサーバーが混雑しています。時間を置いて再度お試しください。({e2})"}
        else:
            return {"success": False, "message": f"解析エラーが発生しました。({e})"}
