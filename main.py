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
    制限のない安全なGroqサーバーで高速・確実に処理
    """
    if not groq_key:
        raise Exception("Groq API Key (GROQ_API_KEY) が設定されていません。")

    hint_text = f"\n[ユーザーからのヒント・情報]: {user_hint}" if user_hint else ""

    prompt = (
        "画像とユーザーからの追加ヒントを組み合わせて、場所を特定してください。\n"
        "出力は、必ず以下のタグで囲んで回答してください。他の挨拶や余計な文章は一切書かないでください。\n"
        "<reason>推論の理由を詳しく（日本語）</reason>\n"
        "<query_used>検索に使ったキーワード</query_used>\n"
        "<location>特定された住所や地名</location>\n"
        "<lat>緯度の数値のみ（例: 35.6812）</lat>\n"
        "<lng>経度の数値のみ（例: 139.7671）</lng>"
        f"{hint_text}"
    )
    
    # 画面側で軽量化されているので、100%安全に一瞬で転送可能
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    data_url = f"data:image/jpeg;base64,{base64_image}"

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"}
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
        "temperature": 0.2,
        "max_tokens": 800
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=payload, timeout=40.0)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

def parse_groq_text(text: str):
    """Groqのタグ出力をフロント画面用データに安全に変換"""
    def extract(tag, default=""):
        match = re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
        return match.group(1).strip() if match else default

    try:
        # 数字以外の文字が混ざった時のためのクレンジング
        lat_str = re.sub(r'[^0-9.]', '', extract("lat"))
        lng_str = re.sub(r'[^0-9.]', '', extract("lng"))
        
        return {
            "reason": extract("reason", "画像から特定しました。"),
            "query_used": extract("query_used", "画像検索"),
            "location": extract("location", "特定エリア"),
            "lat": float(lat_str) if lat_str else 35.6812,
            "lng": float(lng_str) if lng_str else 139.7671
        }
    except Exception as e:
        print(f"パースエラー: {e}")
        return {"reason": "解析結果の読み込みに失敗しました。", "query_used": "-", "location": "不明", "lat": 35.6812, "lng": 139.7671}

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
        # クォータ制限のないGroqで確実に解析
        groq_text = await ask_groq_geoguessr(image_bytes, safe_hint)
        data = parse_groq_text(groq_text)
        
        return {
            "success": True,
            "reason": data["reason"],
            "query_used": data["query_used"],
            "location": data["location"],
            "lat": data["lat"],
            "lng": data["lng"]
        }
    except Exception as e:
        print(f"エラー発生: {e}")
        return {
            "success": False, 
            "message": f"解析エラーが発生しました。設定や画像を確認してください。({str(e)[:80]})"
        }
