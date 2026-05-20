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

# 📂 Renderのサーバー環境でも確実にフォルダを見つけられる記述に変更
templates = Jinja2Templates(directory="templates")

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
    
    try:
        response = await gemini_client.aio.models.generate_content(
            model='gemini-2.5-flash',
            contents=[genai.types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg'), PROMPT]
        )
        return {"status": "success", "text": response.text}
    except Exception as e:
        error_msg = str(e)
        if "503" in error_msg or "demand" in error_msg.lower() or "unavailable" in error_msg.lower():
            print("Gemini 2.5混雑のため、1.5-proに切り替えます...")
            try:
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
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    
    ai_res = await ask_gemini_for_query(image_bytes)
    if ai_res["status"] == "error":
        return {"success": False, "message": ai_res["message"]}
    
    import json
    try:
        ai_data = json.loads(ai_res["text"])
        reason = ai_data.get("reason", "理由なし")
        search_query = ai_data.get("search_query", "")
    except:
        return {"success": False, "message": "AIが正しいデータ形式で返答できませんでした。もう一度お試しください。"}
    
    map_res = await search_map(search_query)
    
    if map_res["found"]:
        return {
            "success": True,
            "reason": reason,
            "query_used": search_query,
            "location": map_res["location"],
            "lat": map_res["lat"],
            "lng": map_res["lng"]
        }
    else:
        return {"success": False, "message": map_res["message"]}
