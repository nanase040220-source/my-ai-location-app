import asyncio
import os
import base64
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from google import genai

app = FastAPI()

# フォルダ階層の設定
current_dir = os.path.dirname(os.path.abspath(__file__))
templates_dir = os.path.join(current_dir, "templates")
templates = Jinja2Templates(directory=templates_dir)

# 環境変数の取得（Geminiのみ）
gemini_key = os.environ.get("GEMINI_API_KEY")

try:
    if gemini_key:
        gemini_client = genai.Client(api_key=gemini_key)
    else:
        gemini_client = None
except Exception as e:
    print(f"Gemini Client Init Error: {e}")
    gemini_client = None

PROMPT = (
    "この写真が撮影された場所（国、都市、ランドマーク名）を推測し、"
    "その根拠と推定される緯度・経度を必ず以下のJSON形式でのみ出力してください。\n"
    "{\"location\": \"国名や都市名\", \"lat\": 緯度(数値), \"lng\": 経度(数値), \"reason\": \"推測した理由\"}"
)

async def ask_gemini(image_bytes: bytes):
    if not gemini_client:
        return {"status": "error", "message": "Renderの環境変数『GEMINI_API_KEY』が正しく設定されていないか、プログラムに届いていません。"}
    try:
        # ⭕ 新しいgoogle-genaiライブラリで画像バイナリを正しく送る形式に修正
        response = await gemini_client.aio.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                genai.types.Part.from_bytes(
                    data=image_bytes,
                    mime_type='image/jpeg',
                ),
                PROMPT
            ]
        )
        return {"status": "success", "data": response.text}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ルーティング（画面表示）
@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def read_index(request: Request):
    if request.method == "HEAD":
        return HTMLResponse(content="", status_code=200)
    return templates.TemplateResponse(request=request, name="index.html")

# 解析エンドポイント（Geminiのみ実行）
@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    gemini_res = await ask_gemini(image_bytes)
    
    # 画面側が壊れないように、OpenAI側にはダミーのエラーメッセージを返しておく
    return {
        "gemini": gemini_res, 
        "openai": {"status": "error", "message": "無料版のためOpenAIの予測はオフにしています。"}
    }
