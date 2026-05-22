import asyncio
import os
import json
import base64
import re
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import httpx

app = FastAPI()
templates = Jinja2Templates(directory="templates")

groq_key = os.environ.get("GROQ_API_KEY")

async def ask_groq_geoguessr(image_bytes: bytes, user_hint: str):
    """
    Groqの厳格な仕様に完全に適合させた画像解析処理（JSONモード強制）
    """
    if not groq_key:
        raise Exception("Groq API Key (GROQ_API_KEY) が設定されていません。")

    hint_text = f"\n[ユーザーからのヒント・情報]: {user_hint}" if user_hint else ""

    prompt = (
        "画像とユーザーからの追加ヒントを組み合わせて、場所を特定してください。\n"
        "回答は、必ず以下のキーを持った一つのJSONオブジェクトとして出力してください。他の挨拶や説明は一切含めないでください。\n"
        "{\n"
        "  \"reason\": \"推論の理由を詳しく（日本語）\",\n"
        "  \"query_used\": \"検索に使ったキーワード\",\n"
        "  \"location\": \"特定された住所や地名\",\n"
        "  \"lat\": 緯度の数値(float),\n"
        "  \"lng\": 経度の数値(float)\n"
        "}"
        f"{hint_text}"
    )
    
    # 🚀 改行コードを完全に排除して完璧なBase64文字列を作成（400エラー対策）
    b64_data = base64.b64encode(image_bytes).decode('utf-8').replace('\n', '').replace('\r', '')
    data_url = f"data:image/jpeg;base64,{b64_data}"

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {groq_key}", 
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "llama-3.2-11b-vision-preview",
        "messages": [
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": prompt}, 
                    {"type": "image_url", "image_url": {"url": data_url}}
                ]
            }
        ],
        # 🚀 GroqにJSONでの返却を絶対強制する設定
        "response_format": {"type": "json_object"},
        "temperature": 0.2,
        "max_tokens": 1024
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=payload, timeout=40.0)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def read_index(request: Request):
    if request.method == "HEAD":
        return HTMLResponse(content="", status_code=200)
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...), hint: str = Form(None)):
    image_bytes = await file.read()
    safe_hint = hint if hint else ""

    try:
        # Groqで解析を実行
        groq_json_res = await ask_groq_geoguessr(image_bytes, safe_hint)
        
        # 強制JSONモードなので、そのままパース可能
        data = json.loads(groq_json_res)
        
        return {
            "success": True,
            "reason": data.get("reason", "画像から特定しました。"),
            "query_used": data.get("query_used", "画像検索"),
            "location": data.get("location", "特定エリア"),
            "lat": float(data.get("lat", 35.6812)),
            "lng": float(data.get("lng", 139.7671))
        }
    except Exception as e:
        print(f"エラー発生: {e}")
        return {
            "success": False, 
            "message": f"解析エラーが発生しました。設定や画像を確認してください。({str(e)[:80]})"
        }
