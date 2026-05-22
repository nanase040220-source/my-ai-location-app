import asyncio
import os
import json
import base64
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
import httpx

app = FastAPI()
templates = Jinja2Templates(directory="templates")

gemini_key = os.environ.get("GEMINI_API_KEY")
groq_key = os.environ.get("GROQ_API_KEY")

try:
    if gemini_key:
        gemini_client = genai.Client(api_key=gemini_key)
    else:
        gemini_client = None
except Exception as e:
    print(f"Gemini Init Error: {e}")
    gemini_client = None

# データ構造の定義（Pydanticモデル）
class AnalysisDetail(BaseModel):
    topography: str = Field(description="地形の分析結果")
    soil: str = Field(description="土壌・地質の分析結果")
    vegetation: str = Field(description="植生・植物の分析結果")
    architecture: str = Field(description="建築様式・インフラの分析結果")
    shadows: str = Field(description="太陽の光と影の角度・方位の分析結果")

class LocationCandidate(BaseModel):
    location: str = Field(description="候補の場所・都市・ランドマーク名")
    probability: str = Field(description="確率（例: 85%）")
    lat: float = Field(description="緯度")
    lng: float = Field(description="経度")

class GeoguessrResponseSchema(BaseModel):
    analysis: AnalysisDetail
    reasoning_logic: str
    candidates: list[LocationCandidate]

GEOGRAPHIC_PROMPT = (
    "You are an expert geoguessr and geographic investigator. Your task is to pinpoint the location of this image by analyzing every single clue systematically.\n\n"
    "STEP 1: Analyze the following elements in extreme detail:\n"
    "- Topography & Landforms (Mountains, plains, valley shapes, coastal features)\n"
    "- Soil & Geology (Color of earth, rock types, sand, asphalt quality)\n"
    "- Vegetation & Flora (Tree species, agricultural crops, climate-specific plants, dryness)\n"
    "- Architectural Style (Building materials, roof shapes, infrastructure, utility poles, license plates, road markings)\n"
    "- Sun & Shadows (Estimate the sun's angle and direction to determine the approximate latitude or hemisphere if possible)\n\n"
    "STEP 2: Combine these observations with the user's optional text hint to cross-reference global regions, countries, or specific prefectures.\n\n"
    "STEP 3: Output your final deduction strictly following the schema structure."
)

async def ask_gemini_geoguessr(image_bytes: bytes, user_hint: str):
    """メインAI: Google Gemini 2.5 Flash（混雑時の自動リトライ機能付き）"""
    if not gemini_client:
        raise Exception("Gemini API Key is missing.")
    
    hint_text = f"\n[USER HINT]: {user_hint}\n" if user_hint else ""
    full_prompt = GEOGRAPHIC_PROMPT + hint_text

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=GeoguessrResponseSchema,
        temperature=0.2
    )
    
    # 🔥 503混雑エラー対策：最大3回まで自動リトライするループ
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = await gemini_client.aio.models.generate_content(
                model='gemini-2.5-flash',
                contents=[types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg'), full_prompt],
                config=config
            )
            return response.text
        except Exception as e:
            # エラーメッセージに503やUNAVAILABLEが含まれる場合は混雑と判断
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                if attempt < max_retries - 1:
                    print(f"Geminiサーバー混雑(503)を検知。2秒後に自動リトライします... ({attempt + 1}/{max_retries})")
                    await asyncio.sleep(2)
                    continue
            raise e

async def ask_groq_geoguessr(image_bytes: bytes, user_hint: str):
    """バックアップAI: Groq Cloud (Llama 3.2 11b Vision)"""
    if not groq_key:
        raise Exception("Groq API Key is missing.")
    
    hint_text = f"\n[USER HINT]: {user_hint}\n" if user_hint else ""
    # JSONのスキーマ構造をテキストで強く指示
    schema_instruction = "\n\nYou MUST respond ONLY in JSON matching this structure: {\"analysis\": {\"topography\":\"...\",\"soil\":\"...\",\"vegetation\":\"...\",\"architecture\":\"...\",\"shadows\":\"...\"}, \"reasoning_logic\":\"...\", \"candidates\": [{\"location\":\"...\",\"probability\":\"...\",\"lat\":0.0,\"lng\":0.0}]}"
    full_prompt = GEOGRAPHIC_PROMPT + hint_text + schema_instruction
    
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    data_url = f"data:image/jpeg;base64,{base64_image}"
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"}
    payload = {
        "model": "llama-3.2-11b-vision-preview",
        "messages": [{"role": "user", "content": [{"type": "text", "text": full_prompt}, {"type": "image_url", "image_url": {"url": data_url}}]}],
        "response_format": {"type": "json_object"},
        "temperature": 0.2
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=payload, timeout=25.0)
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
    
    ai_text_result = None
    used_backup = False
    
    try:
        # 1. まずメインAI（リトライ付き）を試す
        ai_text_result = await ask_gemini_geoguessr(image_bytes, safe_hint)
    except Exception as e:
        print(f"Geminiエラー (リトライ失敗): {e}")
        # 2. ダメならバックアップAI（Groq）を試す
        if groq_key:
            try:
                print("メインAI混雑のため、バックアップAI(Groq)に切り替えます。")
                ai_text_result = await ask_groq_geoguessr(image_bytes, safe_hint)
                used_backup = True
            except Exception as e2:
                return {"success": False, "message": f"AIサーバーが非常に混雑しています。時間を置いて再度お試しください。({e2})"}
        else:
            return {"success": False, "message": "現在Google AIが非常に混雑しています。数分後に再度試すか、RenderにGROQ_API_KEYを登録してください。"}
    
    try:
        # データの解析と安全なバリデーション
        raw_data = json.loads(ai_text_result)
        validated_data = GeoguessrResponseSchema.model_validate(raw_data)
        
        logic = validated_data.reasoning_logic
        if used_backup:
            logic += "\n（※メインAI混雑のため、バックアップAIによる推論結果です）"
            
        return {
            "success": True,
            "analysis": validated_data.analysis.model_dump(),
            "reasoning_logic": logic,
            "candidates": [c.model_dump() for c in validated_data.candidates]
        }
    except Exception as parse_err:
        print(f"JSONパースエラー: {parse_err}")
        return {"success": False, "message": "AIデータの形式エラーが発生しました。もう一度実行してみてください。"}
