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

GEOGRAPHIC_PROMPT = (
    "You are an expert geoguessr and geographic investigator. Your task is to pinpoint the location of this image by analyzing every single clue systematically.\n\n"
    "STEP 1: Analyze the following elements in extreme detail:\n"
    "- Topography & Landforms (Mountains, plains, valley shapes, coastal features)\n"
    "- Soil & Geology (Color of earth, rock types, sand, asphalt quality)\n"
    "- Vegetation & Flora (Tree species, agricultural crops, climate-specific plants, dryness)\n"
    "- Architectural Style (Building materials, roof shapes, infrastructure, utility poles, license plates, road markings)\n"
    "- Sun & Shadows (Estimate the sun's angle and direction to determine the approximate latitude or hemisphere if possible)\n\n"
    "STEP 2: Combine these observations with the user's optional text hint to cross-reference global regions, countries, or specific prefectures.\n\n"
    "STEP 3: Output your final deduction strictly following the JSON structure."
)

async def ask_gemini_geoguessr(image_bytes: bytes, mime_type: str, user_hint: str):
    """メインAI: Google Gemini（動的MIMEタイプ＆JSONモード）"""
    if not gemini_client:
        raise Exception("Gemini API Key is missing.")
    
    hint_text = f"\n[USER HINT]: {user_hint}\n" if user_hint else ""
    
    json_instruction = (
        "\n\nYou MUST respond in JSON format matching this schema exactly:\n"
        "{\n"
        "  \"analysis\": {\n"
        "    \"topography\": \"string\",\n"
        "    \"soil\": \"string\",\n"
        "    \"vegetation\": \"string\",\n"
        "    \"architecture\": \"string\",\n"
        "    \"shadows\": \"string\"\n"
        "  },\n"
        "  \"reasoning_logic\": \"string\",\n"
        "  \"candidates\": [\n"
        "    {\n"
        "      \"location\": \"string\",\n"
        "      \"probability\": \"string\",\n"
        "      \"lat\": float,\n"
        "      \"lng\": float\n"
        "    }\n"
        "  ]\n"
        "}"
    )
    full_prompt = GEOGRAPHIC_PROMPT + hint_text + json_instruction

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=0.2
    )
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = await gemini_client.aio.models.generate_content(
                model='gemini-2.0-flash',
                contents=[types.Part.from_bytes(data=image_bytes, mime_type=mime_type), full_prompt],
                config=config
            )
            return response.text
        except Exception as e:
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    print(f"Gemini混雑(503)を検知。{wait_time}秒後に再試行します... ({attempt + 1}/{max_retries})")
                    await asyncio.sleep(wait_time)
                    continue
            raise e

async def ask_groq_geoguessr(image_bytes: bytes, user_hint: str):
    """バックアップAI: Groq Cloud"""
    if not groq_key:
        raise Exception("Groq API Key is missing.")
    
    hint_text = f"\n[USER HINT]: {user_hint}\n" if user_hint else ""
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
    
    # 🩹 AI側でエラーにならないよう、MIMEタイプを主要な4種に安全にフォールバック
    raw_mime = file.content_type or "image/jpeg"
    mime_type = raw_mime if raw_mime in ["image/jpeg", "image/png", "image/webp", "image/heic"] else "image/jpeg"
    
    ai_text_result = None
    used_backup = False
    
    try:
        ai_text_result = await ask_gemini_geoguessr(image_bytes, mime_type, safe_hint)
    except Exception as e:
        print(f"Geminiエラー: {e}")
        if groq_key:
            try:
                print("メインAI混雑のため、バックアップAI(Groq)に切り替えます。")
                ai_text_result = await ask_groq_geoguessr(image_bytes, safe_hint)
                used_backup = True
            except Exception as e2:
                return {"success": False, "message": f"AIサーバーが混雑しています。時間を置いて再度お試しください。({e2})"}
        else:
            return {"success": False, "message": f"解析エラー(503/400)が発生しました。時間を置くか、RenderにGROQ_API_KEYを登録してください。詳細: {e}"}
    
    try:
        ai_data = json.loads(ai_text_result)
        analysis_dict = ai_data.get("analysis", {})
        
        logic = ai_data.get("reasoning_logic", "分析ロジックの取得に失敗しました。")
        if used_backup:
            logic += "\n（※メインAI混雑のため、バックアップAIによる推論結果です）"
            
        return {
            "success": True,
            "analysis": {
                "topography": analysis_dict.get("topography", "分析なし"),
                "soil": analysis_dict.get("soil", "分析なし"),
                "vegetation": analysis_dict.get("vegetation", "分析なし"),
                "architecture": analysis_dict.get("architecture", "分析なし"),
                "shadows": analysis_dict.get("shadows", "分析なし")
            },
            "reasoning_logic": logic,
            "candidates": ai_data.get("candidates", [])
        }
    except Exception as parse_err:
        print(f"JSONパースエラー: {parse_err}")
        return {"success": False, "message": "AIデータの形式エラーが発生しました。もう一度実行してみてください。"}
