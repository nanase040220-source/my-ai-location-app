import asyncio
import os
import base64
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from google import genai
from openai import AsyncOpenAI

app = FastAPI()

# フォルダ階層のズレ対策
current_dir = os.path.dirname(os.path.abspath(__file__))
templates_dir = os.path.join(current_dir, "templates")
templates = Jinja2Templates(directory=templates_dir)

# 環境変数の取得
gemini_key = os.environ.get("GEMINI_API_KEY")
openai_key = os.environ.get("OPENAI_API_KEY")

# キーが空の場合でも即死（Crash）しないように安全に初期化する
try:
    if gemini_key:
        gemini_client = genai.Client(api_key=gemini_key)
    else:
        gemini_client = None
except Exception as e:
    print(f"Gemini Client Init Error: {e}")
    gemini_client = None

try:
    if openai_key:
        openai_client = AsyncOpenAI(api_key=openai_key)
    else:
        openai_client = None
except Exception as e:
    print(f"OpenAI Client Init Error: {e}")
    openai_client = None

PROMPT = (
    "この写真が撮影された場所（国、都市、ランドマーク名）を推測し、"
    "その根拠と推定される緯度・経度を必ず以下のJSON形式でのみ出力してください。\n"
    "{\"location\": \"国名や都市名\", \"lat\": 緯度(数値), \"lng\": 経度(数値), \"reason\": \"推測した理由\"}"
)

async def ask_gemini(image_bytes: bytes):
    if not gemini_client:
        return {"status": "error", "message": "Renderの環境変数『GEMINI_API_KEY』が正しく設定されていないか、プログラムに届いていません。"}
    try:
        response = await gemini_client.aio.models.generate_content(
            model='gemini-2.5-flash',
            contents=[image_bytes, PROMPT]
        )
        return {"status": "success", "data": response.text}
    except Exception as e:
        return {"status": "error", "message": str(e)}

async def ask_openai(image_bytes: bytes):
    if not openai_client:
        return {"status": "error", "message": "Renderの環境変数『OPENAI_API_KEY』が正しく設定されていないか、プログラムに届いていません。"}
    try:
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ],
                }
            ],
            response_format={"type": "json_object"}
        )
        return {"status": "success", "data": response.choices[0].message.content}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# 📝 これに書き換えます
@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    
    gemini_res, openai_res = await asyncio.gather(
        ask_gemini(image_bytes),
        ask_openai(image_bytes)
    )
    return {"gemini": gemini_res, "openai": openai_res}
