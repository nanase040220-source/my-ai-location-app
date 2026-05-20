import asyncio
import os
import urllib.parse
import json
import base64
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from google import genai
import httpx

app = FastAPI()

# 📂 Renderのサーバー環境でも確実にフォルダを見つけられる記述
templates = Jinja2Templates(directory="templates")

# 🔑 各種APIキーの取得
gemini_key = os.environ.get("GEMINI_API_KEY")
groq_key = os.environ.get("GROQ_API_KEY")  # 混雑時・無料枠制限時のバックアップ用

# Google Gemini クライアント初期化
try:
    if gemini_key:
        gemini_client = genai.Client(api_key=gemini_key)
    else:
        gemini_client = None
except Exception as e:
    print(f"Gemini Init Error: {e}")
    gemini_client = None

# AIに「検索キーワード」を作らせるためのプロンプト
PROMPT = (
    "Analyze this image and guess the location. "
    "Provide a search query (3-5 words, e.g., 'Eiffel Tower Paris') to find this exact place on a map. "
    "You MUST respond ONLY in the following JSON format:\n"
    "{\"reason\": \"推測した理由を日本語で詳細に\", \"search_query\": \"landmark name city country\"}"
)

async def ask_gemini_for_query(image_bytes: bytes):
    """メインAI: Google Gemini 2.5 Flash (JSON構造化出力強制)"""
    if not gemini_client:
        raise Exception("Gemini API Key is missing.")
    
    # 応答を確実にJSONにするための設定
    config = genai.types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema={
            "type": "OBJECT",
            "properties": {
                "reason": {"type": "STRING"},
                "search_query": {"type": "STRING"}
            },
            "required": ["reason", "search_query"]
        }
    )
    
    response = await gemini_client.aio.models.generate_content(
        model='gemini-2.5-flash',
        contents=[genai.types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg'), PROMPT],
        config=config
    )
    return response.text

async def ask_groq_for_query(image_bytes: bytes):
    """バックアップAI: Groq Cloud - Llama 3.2 11b Vision (完全別サーバー・無料)"""
    if not groq_key:
        raise Exception("Groq API Key is missing.")
    
    # 画像をGroqが受け付ける形式(Base64)に変換
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    data_url = f"data:image/jpeg;base64,{base64_image}"
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {groq_key}",
        "Content-Type": "application
