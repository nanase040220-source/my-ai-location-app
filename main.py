import asyncio
import os
import json
import base64
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from google import genai
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

# 🔥 地理・地形・天文学的特徴を徹底分析させるための超強力プロンプト
GEOGRAPHIC_PROMPT = (
    "You are an expert geoguessr and geographic investigator. Your task is to pinpoint the location of this image by analyzing every single clue systematically.\n\n"
    "STEP 1: Analyze the following elements in extreme detail:\n"
    "- Topography & Landforms (Mountains, plains, valley shapes, coastal features)\n"
    "- Soil & Geology (Color of earth, rock types, sand, asphalt quality)\n"
    "- Vegetation & Flora (Tree species, agricultural crops, climate-specific plants, dryness)\n"
    "- Architectural Style (Building materials, roof shapes, infrastructure, utility poles, license plates, road markings)\n"
    "- Sun & Shadows (Estimate the sun's angle and direction to determine the approximate latitude or hemisphere if possible)\n\n"
    "STEP 2: Combine these observations with the user's optional text hint to cross-reference global regions, countries, or specific prefectures.\n\n"
    "STEP 3: Output your final deduction. You MUST respond ONLY in the following strict JSON format. No conversational filler, no markdown wrappers outside JSON:\n"
    "{\n"
    "  \"analysis\": {\n"
    "    \"topography\": \"地形の分析結果（日本語）\",\n"
    "    \"soil\": \"土壌・地質の分析結果（日本語）\",\n"
    "    \"vegetation\": \"植生・植物の分析結果（日本語）\",\n"
    "    \"architecture\": \"建築様式・インフラの分析結果（日本語）\",\n"
    "    \"shadows\": \"太陽の光と影の角度・方位の分析結果（日本語）\"\n"
    "  },\n"
    "  \"reasoning_logic\": \"これらを総合して、なぜその結論に至ったかの大まかな地理的・論理的考察（日本語）\",\n"
    "  \"candidates\": [\n"
    "    {\n"
    "      \"location\": \"第1候補の具体的な場所・都市・ランドマーク名（日本語）\",\n"
    "      \"probability\": \"85%\",\n"
    "      \"lat\": 35.6586,\n"
    "      \"lng\": 139.7454\n"
    "    },\n"
    "    {\n"
    "      \"location\": \"第2候補の具体的な場所・都市・ランドマーク名（日本語）\",\n"
    "      \"probability\": \"40%\",\n"
    "      \"lat\": 35.6605,\n"
    "      \"lng\": 139.7291\n"
    "    },\n"
    "    {\n"
    "      \"location\": \"第3候補の具体的な場所・都市・ランドマーク名（日本語）\",\n"
    "      \"probability\": \"15%\",\n"
    "      \"lat\": 35.7101,\n"
    "      \"lng\": 139.8107\n"
    "    }\n"
    "  ]\n"
    "}"
)

async def ask_gemini_geoguessr(image_bytes: bytes, user_hint: str):
    if not gemini_client:
        raise Exception("Gemini API Key is missing.")
    
    # ユーザーのヒントをプロンプトに動的挿入
    hint_text = f"\n[USER HINT]: {user_hint}\n" if user_hint else ""
    full_prompt = GEOGRAPHIC_PROMPT + hint_text

    config = genai.types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema={
            "type": "OBJECT",
            "properties": {
                "analysis": {
                    "type": "OBJECT",
                    "properties": {
                        "topography": {"type": "STRING"},
                        "soil": {"type": "STRING"},
                        "vegetation": {"type": "STRING"},
                        "architecture": {"type": "STRING"},
                        "shadows": {"type": "STRING"}
                    },
                    "required": ["topography", "soil", "vegetation", "architecture", "shadows"]
                },
                "reasoning_logic": {"type": "STRING"},
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
            "required": ["analysis", "reasoning_logic", "candidates"]
        }
    )
    
    response = await gemini_client.aio.models.generate_content(
        model='gemini-2.5-flash',
        contents=[genai.types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg'), full_prompt],
        config=config
    )
    return response.text

@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def read_index(request: Request):
    if request.method == "HEAD":
        return HTMLResponse(content="", status_code=200)
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...), hint: str = Form("")):
    image_bytes = await file.read()
    
    try:
        ai_text_result = await ask_gemini_geoguessr(image_bytes, hint)
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
