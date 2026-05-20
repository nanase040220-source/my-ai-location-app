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

# 🔥 精度向上のための新しいプロンプト（指示をより具体的に）
PROMPT = (
    "Analyze this image and pinpoint the exact location or landmark.\n"
    "Create the best possible map search query. It should be specific (e.g., 'Tokyo Skytree' instead of just 'Skytree'). "
    "If it is a specific building, monument, or natural feature, use its official name.\n\n"
    "You MUST respond ONLY in the following JSON format. Do not include any other text:\n"
    "{\n"
    "  \"reason\": \"画像に写っている特徴（建物、看板、景色など）から、なぜその場所だと判断したのかのロジックを日本語で詳細に説明してください。\",\n"
    "  \"search_query\": \"Official Landmark Name, City, Country\"\n"
    "}"
)

async def ask_gemini_for_query(image_bytes: bytes):
    """メインAI: Google Gemini 2.5 Flash (JSON構造化)"""
    if not gemini_client:
        raise Exception("Gemini API Key is missing.")
    
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

async def search_map(query: str):
    """地図検索エンジン（精度を高めるためにクエリ調整機能を追加）"""
    # AIが返してきたクエリから不要な記号を削る
    clean_query = query.replace('"', '').replace('"', '').strip()
    encoded_query = urllib.parse.quote(clean_query)
    
    url = f"https://photon.komoot.io/api/?q={encoded_query}&limit=3"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept-Language": "ja,en;q=0.9"
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, timeout=10.0)
            if response.status_code != 200:
                return {"found": False, "message": "地図サーバーが応答しませんでした。"}
            
            data = response.json()
            features = data.get("features", [])
            
            if features:
                # 複数ヒットした場合は、最も信頼度の高そうな最初の候補を採用
                first_result = features[0]
                geometry = first_result.get("geometry", {})
                coordinates = geometry.get("coordinates", [0.0, 0.0])
                properties = first_result.get("properties", {})
                
                # ユーザーに見せる住所情報をきれいに整形
                name = properties.get("name", "")
                city = properties.get("city", properties.get("state", ""))
                country = properties.get("country", "")
                
                parts = [p for p in [country, city, name] if p]
                display_name = " ".join(parts)
                
                return {
                    "found": True,
                    "location": display_name if display_name else clean_query,
                    "lat": float(coordinates[1]),
                    "lng": float(coordinates[0])
                }
            
            # 1度目でヒットしなかった場合、カンマで区切られた後ろの文字（国名など）を削って再検索（セカンドチャンス）
            if "," in clean_query:
                short_query = clean_query.split(",")[0].strip()
                return await search_map(short_query)
                
            return {"found": False, "message": f"地図上で '{clean_query}' の詳細位置が特定できませんでした。"}
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
            
    try:
        ai_data = json.loads(ai_text_result)
        reason = ai_data.get("reason", "理由なし")
        search_query = ai_data.get("search_query", "")
        
        if used_backup:
            reason += "（※バックアップAIによる予測結果です）"
    except Exception as parse_err:
        return {"success": False, "message": "AIデータの形式エラーが発生しました。"}
    
    # 強化した地図検索を実行
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
