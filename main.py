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
yahoo_client_id = os.environ.get("YAHOO_CLIENT_ID")  # 🔥 追加：Yahoo!のアプリケーションID

# Google Gemini クライアント初期化
try:
    if gemini_key:
        gemini_client = genai.Client(api_key=gemini_key)
    else:
        gemini_client = None
except Exception as e:
    print(f"Gemini Init Error: {e}")
    gemini_client = None

# 🔥 精度向上のためのプロンプト
PROMPT = (
    "Analyze this image and pinpoint the exact location or landmark.\n"
    "Create the best possible map search query. It should be specific (e.g., 'Tokyo Skytree' instead of just 'Skytree'). "
    "If it is a specific building, monument, or natural feature, use its official name.\n\n"
    "You MUST respond ONLY in the following JSON format. Do not include any other text:\n"
    "{\n"
    "  \"reason\": \"画像に写っている特徴から、なぜその場所だと判断したのかのロジックを日本語で詳細に説明してください。\",\n"
    "  \"search_query\": \"Official Landmark Name, City, Country\"\n"
    "}"
)

async def ask_gemini_for_query(image_bytes: bytes):
    """メインAI: Google Gemini 2.5 Flash"""
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
    """🔥 高精度：Yahoo! JAPAN Web API (ローカル検索) を使用した日本国内に強い地図検索"""
    clean_query = query.replace('"', '').replace('"', '').strip()
    
    # Yahoo!のキーがない場合は、以前の無料Photon APIで代替検索する安全設計
    if not yahoo_client_id:
        print("YAHOO_CLIENT_IDが設定されていないため、Photon APIで代替検索します。")
        return await search_map_photon_fallback(clean_query)
        
    encoded_query = urllib.parse.quote(clean_query)
    # Yahoo!ローカル検索API (カセット「landmark,address」を指定して高精度化)
    url = f"https://map.yahooapis.jp/search/local/V1/localSearch?appid={yahoo_client_id}&query={encoded_query}&output=json&results=1"
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10.0)
            if response.status_code != 200:
                print(f"Yahoo API エラー: {response.status_code}")
                return await search_map_photon_fallback(clean_query)
                
            data = response.json()
            features = data.get("Feature", [])
            
            if features:
                first_result = features[0]
                name = first_result.get("Name", clean_query)
                geometry = first_result.get("Geometry", {})
                # Yahooの座標は「経度,緯度」の文字列で返ってくるためパースします
                coordinates_str = geometry.get("Coordinates", "0.0,0.0")
                lng_str, lat_str = coordinates_str.split(",")
                
                property_data = first_result.get("Property", {})
                address = property_data.get("Address", "")
                
                display_name = f"{name} ({address})" if address else name
                
                return {
                    "found": True,
                    "location": display_name,
                    "lat": float(lat_str), # 緯度
                    "lng": float(lng_str)  # 経度
                }
            
            # カンマ区切りの後ろを削って再検索を試みる
            if "," in clean_query:
                short_query = clean_query.split(",")[0].strip()
                return await search_map(short_query)
                
            return {"found": False, "message": f"地図上で '{clean_query}' を特定できませんでした。"}
    except Exception as e:
        print(f"Yahoo検索中にエラー: {e}")
        return await search_map_photon_fallback(clean_query)

async def search_map_photon_fallback(query: str):
    """Yahoo!が使えない時のための自動バックアップ検索（Photon）"""
    encoded_query = urllib.parse.quote(query)
    url = f"https://photon.komoot.io/api/?q={encoded_query}&limit=1"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10.0)
            data = response.json()
            features = data.get("features", [])
            if features:
                coords = features[0].get("geometry", {}).get("coordinates", [0.0, 0.0])
                props = features[0].get("properties", {})
                display_name = " ".join([p for p in [props.get("country", ""), props.get("city", ""), props.get("name", "")] if p])
                return {"found": True, "location": display_name if display_name else query, "lat": float(coords[1]), "lng": float(coords[0])}
            return {"found": False, "message": f"地図上で '{query}' が見つかりませんでした。"}
    except:
        return {"found": False, "message": "地図検索サーバーが応答しませんでした。"}

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
    
    # Yahoo!地図検索を実行
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
