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
# 🔍 起動時に環境変数のチェック状態をログに出す（セキュリティのため最初の3文字だけ表示）
print("=== 🛠️ 環境変数 接続診断 ===")
for key in ["GEMINI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY", "CLOUDFLARE_API_TOKEN", "CLOUDFLARE_ACCOUNT_ID"]:
    val = os.environ.get(key)
    if val:
        print(f"✅ {key}: 認識されています (先頭: {val[:3]}...)")
    else:
        print(f"❌ {key}: 見つかりません。設定されていません。")
print("===========================")
templates = Jinja2Templates(directory="templates")

# 各種APIキー・環境変数の取得
gemini_key = os.environ.get("GEMINI_API_KEY")
groq_key = os.environ.get("GROQ_API_KEY")
openrouter_key = os.environ.get("OPENROUTER_API_KEY")
cloudflare_token = os.environ.get("CLOUDFLARE_API_TOKEN")
cloudflare_account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")

# Gemini 初期化
try:
    gemini_client = genai.Client(api_key=gemini_key) if gemini_key else None
except Exception as e:
    print(f"Gemini Init Error: {e}")
    gemini_client = None

def get_prompt(user_hint: str) -> str:
    """すべてのAIで共通使用するJSON強制プロンプト"""
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
    if not gemini_client: raise Exception("Gemini API Key missing")
    full_prompt = get_prompt(user_hint)
    config = types.GenerateContentConfig(response_mime_type="application/json", temperature=0.2)
    
    for attempt in range(retries):
        try:
            print(f"[{model_name}] 試行中... ({attempt + 1}/{retries})")
            response = await gemini_client.aio.models.generate_content(
                model=model_name,
                contents=[types.Part.from_bytes(data=image_bytes, mime_type=mime_type), full_prompt],
                config=config
            )
            return json.loads(response.text)
        except Exception as e:
            print(f"[{model_name}] 失敗: {e}")
            if attempt < retries - 1: await asyncio.sleep(delay)
            else: raise e

async def try_groq(image_bytes: bytes, user_hint: str, retries: int = 3, delay: int = 3):
    if not groq_key: raise Exception("Groq API Key missing")
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
            print(f"[Groq] 試行中... ({attempt + 1}/{retries})")
            async with httpx.AsyncClient() as client:
                response = await client.post(url, headers=headers, json=payload, timeout=40.0)
                response.raise_for_status()
                return json.loads(response.json()["choices"][0]["message"]["content"])
        except Exception as e:
            print(f"[Groq] 失敗: {e}")
            if attempt < retries - 1: await asyncio.sleep(delay)
            else: raise e

async def try_openrouter(image_bytes: bytes, user_hint: str, retries: int = 3, delay: int = 3):
    if not openrouter_key: raise Exception("OpenRouter API Key missing")
    full_prompt = get_prompt(user_hint)
    b64_data = base64.b64encode(image_bytes).decode('utf-8').replace('\n', '').replace('\r', '')
    data_url = f"data:image/jpeg;base64,{b64_data}"

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {openrouter_key}", "Content-Type": "application/json"}
    payload = {
        "model": "google/gemini-2.5-flash:free", # OpenRouter提供の無料Visionモデル
        "messages": [{"role": "user", "content": [{"type": "text", "text": full_prompt}, {"type": "image_url", "image_url": {"url": data_url}}]}],
        "response_format": {"type": "json_object"},
        "temperature": 0.2
    }

    for attempt in range(retries):
        try:
            print(f"[OpenRouter] 試行中... ({attempt + 1}/{retries})")
            async with httpx.AsyncClient() as client:
                response = await client.post(url, headers=headers, json=payload, timeout=40.0)
                response.raise_for_status()
                return json.loads(response.json()["choices"][0]["message"]["content"])
        except Exception as e:
            print(f"[OpenRouter] 失敗: {e}")
            if attempt < retries - 1: await asyncio.sleep(delay)
            else: raise e

async def try_cloudflare(image_bytes: bytes, user_hint: str, retries: int = 3, delay: int = 3):
    if not cloudflare_token or not cloudflare_account_id: raise Exception("Cloudflare Credentials missing")
    full_prompt = get_prompt(user_hint)
    
    # Cloudflare Workers AI (Llava) の画像受取形式に整形
    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
    
    url = f"https://api.cloudflare.com/client/v4/accounts/{cloudflare_account_id}/ai/run/@cf/llava-v1.5-7b-vision-preview"
    headers = {"Authorization": f"Bearer {cloudflare_token}", "Content-Type": "application/json"}
    
    # Cloudflareは画像データを数値配列(int array)として受ける仕様に合わせる
    image_array = list(image_bytes)
    
    payload = {
        "prompt": full_prompt,
        "image": image_array,
        "max_tokens": 512
    }

    for attempt in range(retries):
        try:
            print(f"[Cloudflare] 試行中... ({attempt + 1}/{retries})")
            async with httpx.AsyncClient() as client:
                response = await client.post(url, headers=headers, json=payload, timeout=50.0)
                response.raise_for_status()
                # Cloudflareの戻り値テキストからJSON部分を抽出・パース
                res_text = response.json()["result"]["description"]
                # 万が一余計なテキストが入った場合の安全弁
                json_match = re.search(r'\{.*\}', res_text, re.DOTALL)
                return json.loads(json_match.group(0)) if json_match else json.loads(res_text)
        except Exception as e:
            print(f"[Cloudflare] 失敗: {e}")
            if attempt < retries - 1: await asyncio.sleep(delay)
            else: raise e

@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def read_index(request: Request):
    if request.method == "HEAD": return HTMLResponse(content="", status_code=200)
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...), hint: str = Form(None)):
    image_bytes = await file.read()
    mime_type = file.content_type or "image/jpeg"
    safe_hint = hint if hint else ""

    # 5つのAIが全滅した場合、大ループで最初からもう一周リトライ
    for loop_count in range(2):
        print(f"=== 全体メインループ 第 {loop_count + 1} 周目 ===")
        
        # 1. Gemini Pro
        try:
            data = await try_gemini(image_bytes, mime_type, safe_hint, 'gemini-2.5-pro')
            return {"success": True, "reason": data.get("reason"), "query_used": data.get("query_used"), "location": data.get("location"), "lat": float(data.get("lat")), "lng": float(data.get("lng"))}
        except Exception: print("→ Gemini Pro 失敗。Flashに移行します。")

        # 2. Gemini Free (Flash)
        try:
            data = await try_gemini(image_bytes, mime_type, safe_hint, 'gemini-2.5-flash')
            return {"success": True, "reason": data.get("reason") + " (※Flashで解析)", "query_used": data.get("query_used"), "location": data.get("location"), "lat": float(data.get("lat")), "lng": float(data.get("lng"))}
        except Exception: print("→ Gemini Free 失敗。Groqに移行します。")

        # 3. Groq (Llama)
        try:
            data = await try_groq(image_bytes, safe_hint)
            return {"success": True, "reason": data.get("reason") + " (※Groqで解析)", "query_used": data.get("query_used"), "location": data.get("location"), "lat": float(data.get("lat")), "lng": float(data.get("lng"))}
        except Exception: print("→ Groq 失敗。OpenRouterに移行します。")

        # 4. OpenRouter (Gemini Free経由など)
        try:
            data = await try_openrouter(image_bytes, safe_hint)
            return {"success": True, "reason": data.get("reason") + " (※OpenRouterで解析)", "query_used": data.get("query_used"), "location": data.get("location"), "lat": float(data.get("lat")), "lng": float(data.get("lng"))}
        except Exception: print("→ OpenRouter 失敗。Cloudflareに移行します。")

        # 5. Cloudflare Workers AI
        try:
            data = await try_cloudflare(image_bytes, safe_hint)
            return {"success": True, "reason": data.get("reason") + " (※Cloudflareで解析)", "query_used": data.get("query_used"), "location": data.get("location"), "lat": float(data.get("lat")), "lng": float(data.get("lng"))}
        except Exception:
            print("→ Cloudflare も失敗しました。")
            if loop_count == 0:
                print("5秒後に1つ目のAIから再チャレンジします...")
                await asyncio.sleep(5)

    return {
        "success": False, 
        "message": "すべてのAIエンドポイントおよび自動リトライが制限・混雑により全滅しました。APIキーのクォータや設定をご確認ください。"
    }
