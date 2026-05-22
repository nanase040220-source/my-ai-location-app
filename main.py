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
from pydantic import BaseModel, Field  # 🔥 追加：エラーを絶対に起こさないためのデータ固定機能
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

# 🔥 AIの出力形式を100%エラーなく固定するための定義（Pydanticモデル）
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

# 強力な地理特定プロンプト
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
    if not gemini_client:
        raise Exception("Gemini API Key is missing.")
    
    hint_text = f"\n[USER HINT]: {user_hint}\n" if user_hint else ""
    full_prompt = GEOGRAPHIC_PROMPT + hint_text

    # 🔥 response_schemaにPydanticモデルを指定することで、AIのデータ崩れによるエラーを完全に防ぎます
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=GeoguessrResponseSchema,
        temperature=0.2
    )
    
    response = await gemini_client.aio.models.generate_content(
        model='gemini-2.5-flash',
        contents=[types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg'), full_prompt],
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
    safe_hint = hint if hint else ""
    
    try:
        ai_text_result = await ask_gemini_geoguessr(image_bytes, safe_hint)
        ai_data = json.loads(ai_text_result)
        
        return {
            "success": True,
            "analysis": ai_data.get("analysis"),
            "reasoning_logic": ai_data.get("reasoning_logic"),
            "candidates": ai_data.get("candidates", [])
        }
    except Exception as e:
        print(f"Error: {e}")
        return {"success": False, "message": f"解析中にエラーが発生しました: {str(e)}"}
