import asyncio
import os
import json
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from google import genai
from google.genai import types

app = FastAPI()
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

async def ask_gemini_geoguessr(image_bytes: bytes, mime_type: str, user_hint: str, use_backup_model: bool = False):
    """Google Gemini（メイン＆バックアップ兼用、大容量画像もエラーなしで高速処理）"""
    if not gemini_client:
        raise Exception("Gemini API Key is missing.")

    # 使うモデルを切り替える（1つ目が混雑していたらより強力なプレミアムモデルに自動切替）
    model_name = 'gemini-2.5-pro' if use_backup_model else 'gemini-2.0-flash'

    hint_text = f"\n[ユーザーからの追加情報・ヒント]: {user_hint}\n" if user_hint else ""

    prompt = (
        "画像とユーザーからの追加ヒントを組み合わせて、場所を特定してください。必ず以下のJSONフォーマットのみで返してください。\n"
        "{\n"
        "  \"reason\": \"推論の理由（日本語）\",\n"
        "  \"query_used\": \"検索に使ったキーワード\",\n"
        "  \"location\": \"特定された住所や地名\",\n"
        "  \"lat\": 緯度(float),\n"
        "  \"lng\": 経度(float)\n"
        "}"
    )
    
    full_prompt = prompt + hint_text

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=0.2
    )
    
    response = await gemini_client.aio.models.generate_content(
        model=model_name,
        contents=[types.Part.from_bytes(data=image_bytes, mime_type=mime_type), full_prompt],
        config=config
    )
    return response.text

@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def read_index(request: Request):
    if request.method == "HEAD":
        return HTMLResponse(content="", status_code=200)
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...), hint: str = Form(None)):
    image_bytes = await file.read()
    mime_type = file.content_type or "image/jpeg"
    safe_hint = hint if hint else ""

    try:
        # 1. まずは通常のGeminiモデルで解析
        gemini_result = await ask_gemini_geoguessr(image_bytes, mime_type, safe_hint, use_backup_model=False)
        data = json.loads(gemini_result)
        return {
            "success": True,
            "reason": data.get("reason", "理由なし"),
            "query_used": data.get("query_used", "検索クエリ"),
            "location": data.get("location", "不明"),
            "lat": float(data.get("lat", 0)),
            "lng": float(data.get("lng", 0))
        }
    except Exception as e:
        print(f"メインモデル混雑、バックアップ高級モデルへ移行: {e}")
        try:
            # 2. 失敗した（混雑している）場合は、超高性能なプロモデル（Gemini 2.5 Pro）でリトライ
            gemini_result = await ask_gemini_geoguessr(image_bytes, mime_type, safe_hint, use_backup_model=True)
            data = json.loads(gemini_result)
            return {
                "success": True,
                "reason": data.get("reason", "") + " (※高精度バックアップAIで解析)",
                "query_used": data.get("query_used", ""),
                "location": data.get("location", ""),
                "lat": float(data.get("lat", 0)),
                "lng": float(data.get("lng", 0))
            }
        except Exception as e2:
            return {"success": False, "message": f"AIサーバーが非常に混雑しています。15秒ほど置いて再度お試しください。({e2})"}
