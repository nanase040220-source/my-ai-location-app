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
groq_key = os.environ.get("GROQ_API_KEY")

# Google Gemini クライアント初期化
try:
    if gemini_key:
        gemini_client = genai.Client(api_key=gemini_key)
    else:
        gemini_client = None
except Exception as e:
    print(f"Gemini Init Error: {e}")
    gemini_client = None

# 🔥 AIに「可能性のある候補を最大3つ」と「確率(%)」を出させる新しいプロンプト
PROMPT = (
    "Analyze this image and predict the top 3 most likely locations or landmarks. "
    "For each location, provide an estimated confidence percentage (the total does not need to equal 100%, just your confidence for each). "
    "Also, provide the approximate latitude and longitude for each location.\n\n"
    "You MUST respond ONLY in the following JSON format. Do not include any other text:\n"
    "{\n"
    "  \"reason\": \"画像全体の特徴（建物、気候、看板、植物など）から、候補地を絞り込んだ総合的な推測理由を日本語で\",\n"
    "  \"candidates\": [\n"
    "    {\n"
    "      \"location\": \"第1候補の場所・ランドマーク名（日本語）\",\n"
    "      \"probability\": \"85%\",\n"
    "      \"lat\": 35.6586,\n"
    "      \"lng\": 139.7454\n"
    "    },\n"
    "    {\n"
    "      \"location\": \"第2候補の場所・ランドマーク名（日本語）\",\n"
    "      \"probability\": \"40%\",\n"
    "      \"lat\": 35.6605,\n"
    "      \"lng\": 139.7291\n"
    "    },\n"
    "    {\n"
    "      \"location\": \"第3候補の場所・ランドマーク名（日本語）\",\n"
    "      \"probability\": \"15%\",\n"
    "      \"lat\": 35.7101,\n"
    "      \"lng\": 139.8107\n"
    "    }\n"
    "  ]\n"
    "}"
)

async def ask_gemini_for_query(image_bytes: bytes):
    """メインAI: Google Gemini 2.5 Flash (複数候補JSON構造化)"""
    if not gemini_client:
        raise Exception("Gemini API Key is missing.")
    
    # 応答スキーマを複数候補（配列）対応にアップデート
    config = genai.types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema={
            "type": "OBJECT",
            "properties": {
                "reason": {"type": "STRING"},
                "candidates": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "location": {"type": "STRING"},
                            "probability": {"type": "STRING"},
                            "lat": {"type": "NUMBER"},
                            "lng": {"type": "NUMBER"}
                        },
                        "required": ["location", "probability", "lat", "lng"]
                    }
                }
            },
            "required": ["reason", "candidates"]
        }
    )
    
    response = await gemini_client.aio.models.generate_content(
        model='gemini-2.5-flash',
        contents=[genai.types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg'), PROMPT],
        config=config
    )
    return response.text

async def ask_groq_for_query(image_bytes: bytes):
    """バックアップAI: Groq Cloud - Llama 3.2 11b Vision"""
    if not groq_key:
        raise Exception("Groq API Key is missing.")
    
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    data_url = f"data:image/jpeg;base64,{base64_image}"
    
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
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}}
                ]
            }
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.1
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=payload, timeout=20.0)
        response.raise_for_status()
        res_json = response.json()
        return res_json["choices"][0]["message"]["content"]

@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def read_index(request: Request):
    if request.method == "HEAD":
        return HTMLResponse(content="", status_code=200)
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    
    ai_text_result = None
    used_backup = False
    
    try:
        ai_text_result = await ask_gemini_for_query(image_bytes)
    except Exception as e:
        print(f"Geminiエラー: {e}")
        if groq_key:
            try:
                ai_text_result = await ask_groq_for_query(image_bytes)
                used_backup = True
            except Exception as e2:
                return {"success": False, "message": f"AI提供元が混雑しています。({e2})"}
        else:
            return {"success": False, "message": f"メインAIが混雑しています。バックアップキーを登録してください。"}
            
    # JSONの解析とフロントエンドへの返却
    try:
        ai_data = json.loads(ai_text_result)
        reason = ai_data.get("reason", "理由なし")
        candidates = ai_data.get("candidates", [])
        
        if used_backup:
            reason += "（※バックアップAIによる予測結果です）"
            
        if not candidates:
            return {"success": False, "message": "候補地を見つけられませんでした。"}
            
        # 💡 地図APIを使わず、AIが出した正確な座標付きの複数候補をそのまま返します
        return {
            "success": True,
            "reason": reason,
            "candidates": candidates  # 複数候補のリスト（場所名、確率、緯度、経度を含む）
        }
        
    except Exception as parse_err:
        print(f"JSONパースエラー: {parse_err}")
        return {"success": False, "message": "AIデータの形式エラーが発生しました。"}
