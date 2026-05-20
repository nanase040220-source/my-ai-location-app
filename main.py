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
        "temperature": 0.2
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=payload, timeout=20.0)
        response.raise_for_status()
        res_json = response.json()
        return res_json["choices"][0]["message"]["content"]

async def search_map(query: str):
    """無料かつ制限の緩い Photon 地図検索APIを使用（エラー対策）"""
    encoded_query = urllib.parse.quote(query)
    url = f"https://photon.komoot.io/api/?q={encoded_query}&limit=1"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept-Language": "ja,en;q=0.9"
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, timeout=10.0)
            
            if response.status_code != 200:
                print(f"地図APIエラー。ステータス: {response.status_code}")
                return {"found": False, "message": "地図サーバーが応答しませんでした。"}
            
            data = response.json()
            features = data.get("features", [])
            
            if features:
                first_result = features[0]
                geometry = first_result.get("geometry", {})
                coordinates = geometry.get("coordinates", [0.0, 0.0])
                properties = first_result.get("properties", {})
                
                # 表示用住所の組み立て
                name = properties.get("name", "")
                city = properties.get("city", "")
                country = properties.get("country", "")
                display_name = ", ".join([p for p in [name, city, country] if p])
                
                return {
                    "found": True,
                    "location": display_name if display_name else "不明な場所",
                    "lat": float(coordinates[1]),
                    "lng": float(coordinates[0])
                }
            return {"found": False, "message": f"地図上で '{query}' が見つかりませんでした。"}
    except Exception as e:
        print(f"地図検索中に例外発生: {str(e)}")
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
    
    # 【AI自動切り替えロジック】
    try:
        print("メインAI (Gemini 2.5) で解析中...")
        ai_text_result = await ask_gemini_for_query(image_bytes)
        print("Geminiでの解析に成功しました。")
    except Exception as e:
        print(f"Geminiでエラーまたは混雑が発生しました: {e}")
        
        # Groqのキーが登録されている場合のみバックアップを起動
        if groq_key:
            print("バックアップAI (Groq: Llama3.2 Vision) に切り替えます...")
            try:
                ai_text_result = await ask_groq_for_query(image_bytes)
                used_backup = True
                print("Groqでの代替解析に成功しました！")
            except Exception as e2:
                print(f"バックアップAIでもエラーが発生しました: {e2}")
                return {"success": False, "message": f"すべてのAI提供元が混雑しています。時間をおいて再度お試しください。({e2})"}
        else:
            return {"success": False, "message": f"メインAIが混雑しています。バックアップ用のGROQ_API_KEYが設定されていません。({e})"}
            
    # JSONパース（解析結果の読み込み）
    try:
        ai_data = json.loads(ai_text_result)
        reason = ai_data.get("reason", "理由なし")
        search_query = ai_data.get("search_query", "")
        
        if used_backup:
            reason += "（※バックアップAIによる推測結果です）"
    except Exception as parse_err:
        print(f"JSONパースエラー。生データ: {ai_text_result}, エラー: {parse_err}")
        return {"success": False, "message": "AIデータの読み込みに失敗しました。もう一度お試しください。"}
    
    # 地図検索
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
