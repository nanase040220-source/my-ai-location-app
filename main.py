import asyncio
import os
import urllib.parse
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from google import genai
import httpx

app = FastAPI()

current_dir = os.path.dirname(os.path.abspath(__file__))
templates_dir = os.path.join(current_dir, "templates")
templates = Jinja2Templates(directory=templates_dir)

gemini_key = os.environ.get("GEMINI_API_KEY")

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
    "{\"reason\": \"推測した理由を日本語で\", \"search_query\": \"landmark name city country\"}"
)

async def ask_gemini_for_query(image_bytes: bytes):
    if not gemini_client:
        return {"status": "error", "message": "API Key is missing."}
    
    # 1本目の矢: Gemini 2.5-flash
    try:
        response = await gemini_client.aio.models.generate_content(
            model='gemini-2.5-flash',
            contents=[genai.types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg'), PROMPT]
        )
        return {"status": "success", "text": response.text}
    except Exception as e:
        error_msg = str(e)
        # 混雑エラー（503やunavailableなど）が起きた場合
        if "503" in error_msg or "demand" in error_msg.lower() or "unavailable" in error_msg.lower():
            print("Gemini 2.5混雑のため、1.5-proに切り替えます...")
            try:
                # 2本目の矢: 確実に対応している上位無料モデル gemini-1.5-pro に変更
                response = await gemini_client.aio.models.generate_content(
                    model='gemini-1.5-pro',
                    contents=[genai.types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg'), PROMPT]
                )
                return {"status": "success", "text": response.text}
            except Exception as e2:
                return {"status": "error", "message": f"AI混雑エラー: {e2}"}
        return {"status": "error", "message": error_msg}

# 完全無料の地図データベース（OpenStreetMap）を検索する関数
async def search_map(query: str):
    encoded_query = urllib.parse.quote(query)
    url = f"https://nominatim.openstreetmap.org/search?q={encoded_query}&format=json&limit=1"
    headers = {"User-Agent": "MyAiLocationApp/1.0 (test@example.com)"}
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers)
            data = response.json()
            if data:
                return {
                    "found": True,
                    "location": data[0].get("display_name"),
                    "lat": float(data[0].get("lat")),
                    "lng": float(data[0].get("lon"))
                }
            return {"found": False, "message": f"地図上で '{query}' が見つかりませんでした。"}
    except Exception as e:
        return {"found": False, "message": f"地図検索エラー: {str(e)}"}

@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def read_index(request: Request):
    if request.method == "HEAD":
        return HTMLResponse(content="", status_code=200)
