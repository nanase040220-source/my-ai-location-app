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

# -------------------------------------------------------------
# 修正箇所：フォルダの場所を絶対パスで確実に指定する
# -------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(current_dir, "templates"))

# 環境変数から各AIのAPIキーを取得
gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# AIにフォーマットを固定させるための共通プロンプト
PROMPT = (
    "この写真が撮影された場所（国、都市、ランドマーク名）を推測し、"
    "その根拠と推定される緯度・経度を必ず以下のJSON形式でのみ出力してください。\n"
    "{\"location\": \"国名や都市名\", \"lat\": 緯度(数値), \"lng\": 経度(数値), \"reason\": \"推測した理由\"}"
)

async def ask_gemini(image_bytes: bytes):
    try:
        response = await gemini_client.aio.models.generate_content(
            model='gemini-2.5-flash',
            contents=[image_bytes, PROMPT]
        )
        return {"status": "success", "data": response.text}
    except Exception as e:
        return {"status": "error", "message": str(e)}

async def ask_openai(image_bytes: bytes):
    try:
        # OpenAI用に画像をBase64形式に変換
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
            response_format={"type": "json_object"} # JSON出力を強制
        )
        return {"status": "success", "data": response.choices[0].message.content}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# サイトのトップページ（URLにアクセスされた時にHTML画面を表示する）
@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# 画像解析用エンドポイント
@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    
    # 並列で両方のAIに問い合わせ
    gemini_res, openai_res = await asyncio.gather(
        ask_gemini(image_bytes),
        ask_openai(image_bytes)
    )
    return {"gemini": gemini_res, "openai": openai_res}
