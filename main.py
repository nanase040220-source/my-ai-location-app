import asyncio
import os
import json
import base64
import re
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

# Gemini クライアント初期化
try:
    gemini_client = genai.Client(api_key=gemini_key) if gemini_key else None
except Exception as e:
    print(f"Gemini Init Error: {e}")
    gemini_client = None

def get_prompt(user_hint: str) -> str:
    """共通のプロンプト（JSON形式を強制）"""
    hint_text = f"\n[ユーザーからのヒント・情報]: {user_hint}" if user_hint else ""
    return (
        "画像とユーザーからの追加ヒントを組み合わせて、場所を特定してください。\n"
        "必ず以下のJSONフォーマットのみで返してください。他の挨拶や説明は一切含めないでください。\n"
        "{\n"
        "  \"reason\": \"推論の理由（日本語）\",\n"
        "  \"query_used\": \"検索に使ったキーワード\",\n"
        "  \"location\": \"特定された住所や地名\",\n"
        "  \"lat\": 緯度(float),\n"
        "  \"lng\": 経度(float)\n"
        "}"
        f"{hint_text}"
    )

async def try_gemini(image_bytes: bytes, mime_type: str, user_hint: str, model_name: str, retries: int = 3, delay: int = 3):
    """Geminiモデル用のリトライ機能付き実行関数"""
    if not gemini_client:
        raise Exception("Gemini API Key is missing.")
        
    full_prompt = get_prompt(user_hint)
    config = types.GenerateContentConfig(response_mime_type="application/json", temperature=0.2)

    for attempt in range(retries):
        try:
            print(f"[{model_name}] 解析試行中... ({attempt + 1}/{retries})")
            response = await gemini_client.aio.models.generate_content(
                model=model_name,
                contents=[types.Part.from_bytes(data=image_bytes, mime_type=mime_type), full_prompt],
                config=config
            )
            # 正常に取得できたらJSONをパースして返す
            return json.loads(response.text)
        except Exception as e:
            print(f"[{model_name}] 失敗 (残りリトライ {retries - attempt - 1}): {e}")
            if attempt < retries - 1:
                await asyncio.sleep(delay) # 指定秒数待機してリトライ
            else:
                raise e # リトライを使い切ったら次のAIへ投げるためエラーを出す

async def try_groq(image_bytes: bytes, user_hint: str, retries: int = 3, delay: int = 3):
    """Groq用のリトライ機能付き実行関数（JSONモード強制）"""
    if not groq_key:
        raise Exception("Groq API Key is missing.")

    full_prompt = get_prompt(user_hint)
    b64_data = base64.b64encode(image_bytes).decode('utf-8').replace('\n', '').replace('\r', '')
    data_url = f"data:image/jpeg;base64,{b64_data}"

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"}
    payload = {
        "model": "llama-3.2-11b-vision-preview",
        "messages": [{"role": "user", "content": [{"type": "text", "text": full_prompt}, {"type": "image_url", "image_url": {"url": data_url}}]}],
        "response_format": {"type": "json_object"},
        "temperature": 0.2,
        "max_tokens": 1024
    }

    for attempt in range(retries):
        try:
            print(f"[Groq] 解析試行中... ({attempt + 1}/{retries})")
            async with httpx.AsyncClient() as client:
                response = await client.post(url, headers=headers, json=payload, timeout=40.0)
                response.raise_for_status()
                res_json = response.json()["choices"][0]["message"]["content"]
                return json.loads(res_json)
        except Exception as e:
            print(f"[Groq] 失敗 (残りリトライ {retries - attempt - 1}): {e}")
            if attempt < retries - 1:
                await asyncio.sleep(delay)
            else:
                raise e

@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def read_index(request: Request):
    if request.method == "HEAD":
        return HTMLResponse(content="", status_code=200)
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...), hint: str = Form(None)):
    image_bytes = await file.read()
    mime_type = file.content_type or "image/jpeg"
    safe_hint = hint if hint else ""

    # 全体が全滅した場合、もう一周最初からリトライを繰り返すための大ループ（最大2周）
    for loop_count in range(2):
        print(f"--- 解析メインループ第 {loop_count + 1} 周目 ---")
        
        # 1. Gemini Pro (高級モデル) + リトライ
        try:
            data = await try_gemini(image_bytes, mime_type, safe_hint, model_name='gemini-2.5-pro')
            return {"success": True, "reason": data.get("reason"), "query_used": data.get("query_used"), "location": data.get("location"), "lat": float(data.get("lat")), "lng": float(data.get("lng"))}
        except Exception:
            print("Gemini Pro が失敗しました。Gemini Free (Flash) に切り替えます。")

        # 2. Gemini Free (Flashモデル) + リトライ
        try:
            data = await try_gemini(image_bytes, mime_type, safe_hint, model_name='gemini-2.5-flash')
            return {"success": True, "reason": data.get("reason") + " (※無料枠AIで解析)", "query_used": data.get("query_used"), "location": data.get("location"), "lat": float(data.get("lat")), "lng": float(data.get("lng"))}
        except Exception:
            print("Gemini Free が失敗しました。別AI (Groq) に切り替えます。")

        # 3. 別AI Groq (Llama 3.2 Vision) + リトライ
        try:
            data = await try_groq(image_bytes, safe_hint)
            return {"success": True, "reason": data.get("reason") + " (※別AI Groqで解析)", "query_used": data.get("query_used"), "location": data.get("location"), "lat": float(data.get("lat")), "lng": float(data.get("lng"))}
        except Exception:
            print("Groq も失敗しました。")
            if loop_count == 0:
                print("インターバルを置いて、最初からもう一度チャレンジします。")
                await asyncio.sleep(5) # 次の周に進む前に少し間を置く

    # すべてのAI、すべての自動リトライ、全周回が完全に全滅した場合のみエラー通知
    return {
        "success": False, 
        "message": "すべてのAIサーバーが一時的に制限、または混雑しています。しばらく時間を空けて再度お試しください。"
    }
