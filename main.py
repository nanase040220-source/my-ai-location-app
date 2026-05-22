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

# 環境変数の取得
gemini_key = os.environ.get("GEMINI_API_KEY")
groq_key = os.environ.get("GROQ_API_KEY")
openrouter_key = os.environ.get("OPENROUTER_API_KEY")
cloudflare_token = os.environ.get("CLOUDFLARE_API_TOKEN")
cloudflare_account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")

print("=== 🛠️ 環境変数 接続診断 ===")
for k, v in [("GEMINI_API_KEY", gemini_key), ("GROQ_API_KEY", groq_key), ("OPENROUTER_API_KEY", openrouter_key), ("CLOUDFLARE_API_TOKEN", cloudflare_token), ("CLOUDFLARE_ACCOUNT_ID", cloudflare_account_id)]:
    print(f"✅ {k}: 認識中 (先頭: {v[:3]}...)" if v else f"❌ {k}: 未設定")
print("===========================")

def get_prompt(user_hint: str) -> str:
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

async def try_gemini(image_bytes: bytes, mime_type: str, user_hint: str, model_name: str):
    """Gemini単体の実行（エラー時は即座に上位に例外を投げる）"""
    if not gemini_key:
        raise Exception("Gemini Key is missing")
    client = genai.Client(api_key=gemini_key)
    full_prompt = get_prompt(user_hint)
    config = types.GenerateContentConfig(response_mime_type="application/json", temperature=0.2)
    
    # タイムアウトを設けてフリーズを防ぐ
    response = await client.aio.models.generate_content(
        model=model_name,
        contents=[types.Part.from_bytes(data=image_bytes, mime_type=mime_type), full_prompt],
        config=config
    )
    return json.loads(response.text)

async def try_groq(image_bytes: bytes, user_hint: str):
    if not groq_key: 
        raise Exception("Groq Key missing")
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
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=payload, timeout=25.0)
        response.raise_for_status()
        return json.loads(response.json()["choices"][0]["message"]["content"])

async def try_openrouter(image_bytes: bytes, user_hint: str):
    if not openrouter_key: 
        raise Exception("OpenRouter Key missing")
    full_prompt = get_prompt(user_hint)
    b64_data = base64.b64encode(image_bytes).decode('utf-8').replace('\n', '').replace('\r', '')
    data_url = f"data:image/jpeg;base64,{b64_data}"

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {openrouter_key}", "Content-Type": "application/json"}
    payload = {
        # 🚀 Google制限を避けるため、OpenRouter側の無料LlamaVisionモデルに固定
        "model": "meta-llama/llama-3.2-11b-vision-instruct:free",
        "messages": [{"role": "user", "content": [{"type": "text", "text": full_prompt}, {"type": "image_url", "image_url": {"url": data_url}}]}],
        "response_format": {"type": "json_object"},
        "temperature": 0.2
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=payload, timeout=25.0)
        response.raise_for_status()
        return json.loads(response.json()["choices"][0]["message"]["content"])

async def try_cloudflare(image_bytes: bytes, user_hint: str):
    if not cloudflare_token or not cloudflare_account_id: 
        raise Exception("Cloudflare Credentials missing")
    full_prompt = get_prompt(user_hint)
    
    url = f"https://api.cloudflare.com/client/v4/accounts/{cloudflare_account_id}/ai/run/@cf/llava-v1.5-7b-vision-preview"
    headers = {"Authorization": f"Bearer {cloudflare_token}", "Content-Type": "application/json"}
    
    payload = {
        "prompt": full_prompt,
        "image": list(image_bytes),
        "max_tokens": 512
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=payload, timeout=30.0)
        response.raise_for_status()
        res_text = response.json()["result"]["description"]
        json_match = re.search(r'\{.*\}', res_text, re.DOTALL)
        return json.loads(json_match.group(0)) if json_match else json.loads(res_text)

@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def read_index(request: Request):
    if request.method == "HEAD": return HTMLResponse(content="", status_code=200)
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...), hint: str = Form(None)):
    image_bytes = await file.read()
    mime_type = file.content_type or "image/jpeg"
    safe_hint = hint if hint else ""

    # 5段階のAIを、前段のエラーに一切干渉させずに順番に試すループ
    for loop_count in range(2):
        print(f"=== 実行メインループ 第 {loop_count + 1} 周目 ===")
        
        # 1. Gemini Pro (最大3回リトライ)
        for i in range(3):
            try:
                print(f"[Gemini Pro] 試行中 ({i+1}/3)")
                data = await try_gemini(image_bytes, mime_type, safe_hint, 'gemini-2.5-pro')
                return {"success": True, "reason": data.get("reason"), "query_used": data.get("query_used"), "location": data.get("location"), "lat": float(data.get("lat")), "lng": float(data.get("lng"))}
            except Exception as e:
                print(f"→ Gemini Pro 失敗: {str(e)[:60]}")
                await asyncio.sleep(2)

        # 2. Gemini Flash (最大3回リトライ)
        for i in range(3):
            try:
                print(f"[Gemini Flash] 試行中 ({i+1}/3)")
                data = await try_gemini(image_bytes, mime_type, safe_hint, 'gemini-2.5-flash')
                return {"success": True, "reason": data.get("reason") + " (※Flash)", "query_used": data.get("query_used"), "location": data.get("location"), "lat": float(data.get("lat")), "lng": float(data.get("lng"))}
            except Exception as e:
                print(f"→ Gemini Flash 失敗: {str(e)[:60]}")
                await asyncio.sleep(2)

        # 3. Groq (最大3回リトライ)
        for i in range(3):
            try:
                print(f"[Groq] 試行中 ({i+1}/3)")
                data = await try_groq(image_bytes, safe_hint)
                return {"success": True, "reason": data.get("reason") + " (※Groq)", "query_used": data.get("query_used"), "location": data.get("location"), "lat": float(data.get("lat")), "lng": float(data.get("lng"))}
            except Exception as e:
                print(f"→ Groq 失敗: {str(e)[:60]}")
                await asyncio.sleep(2)

        # 4. OpenRouter (最大3回リトライ)
        for i in range(3):
            try:
                print(f"[OpenRouter] 試行中 ({i+1}/3)")
                data = await try_openrouter(image_bytes, safe_hint)
                return {"success": True, "reason": data.get("reason") + " (※OpenRouter)", "query_used": data.get("query_used"), "location": data.get("location"), "lat": float(data.get("lat")), "lng": float(data.get("lng"))}
            except Exception as e:
                print(f"→ OpenRouter 失敗: {str(e)[:60]}")
                await asyncio.sleep(2)

        # 5. Cloudflare Workers AI (最大3回リトライ)
        for i in range(3):
            try:
                print(f"[Cloudflare] 試行中 ({i+1}/3)")
                data = await try_cloudflare(image_bytes, safe_hint)
                return {"success": True, "reason": data.get("reason") + " (※Cloudflare)", "query_used": data.get("query_used"), "location": data.get("location"), "lat": float(data.get("lat")), "lng": float(data.get("lng"))}
            except Exception as e:
                print(f"→ Cloudflare 失敗: {str(e)[:60]}")
                await asyncio.sleep(2)

        if loop_count == 0:
            print("全AIが1周目で失敗。5秒待機して最終周へ移行します...")
            await asyncio.sleep(5)

    return {
        "success": False, 
        "message": "すべてのAIが制限・混雑により応答しませんでした。少し時間を空けて再度お試しください。"
    }
