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
        "出力は必ず次のJSON文字列のみにしてください。前後の挨拶、コードブロック(```json)、説明は一切含めないでください。\n"
        "{\n"
        "  \"reason\": \"推論の理由（日本語）\",\n"
        "  \"query_used\": \"検索に使ったキーワード\",\n"
        "  \"location\": \"特定された住所や地名\",\n"
        "  \"lat\": 35.6895,\n"
        "  \"lng\": 139.6917\n"
        "}"
        f"{hint_text}"
    )

def extract_json_safe(text: str) -> dict:
    """AIが返した無骨なテキストから無理やりJSON部分だけを抜き出す安全なパーサー"""
    try:
        # まずそのままパースを試みる
        return json.loads(text.strip())
    except Exception:
        # 失敗した場合、テキスト内の最初と最後の { } を探して切り出す
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    # 最悪パースが全て失敗した場合のフォールバック
    return {
        "reason": f"AIの応答のパースに失敗しました。生の応答: {text[:100]}...",
        "query_used": "不明",
        "location": "パースエラーエリア",
        "lat": 35.6895,
        "lng": 139.6917
    }

async def try_groq(image_bytes: bytes, user_hint: str):
    if not groq_key: raise Exception("Groq Key missing")
    full_prompt = get_prompt(user_hint)
    b64_data = base64.b64encode(image_bytes).decode('utf-8')
    
    url = "[https://api.groq.com/openai/v1/chat/completions](https://api.groq.com/openai/v1/chat/completions)"
    headers = {"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"}
    
    # 400エラーの原因になりやすいresponse_formatを敢えて外し、プレーンテキストで回収する
    payload = {
        "model": "llama-3.2-11b-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": full_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_data}"}}
                ]
            }
        ],
        "temperature": 0.2,
        "max_tokens": 1024
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=payload, timeout=20.0)
        response.raise_for_status()
        res_text = response.json()["choices"][0]["message"]["content"]
        return extract_json_safe(res_text)

async def try_openrouter(image_bytes: bytes, user_hint: str):
    if not openrouter_key: raise Exception("OpenRouter Key missing")
    full_prompt = get_prompt(user_hint)
    b64_data = base64.b64encode(image_bytes).decode('utf-8')

    url = "[https://openrouter.ai/api/v1/chat/completions](https://openrouter.ai/api/v1/chat/completions)"
    headers = {
        "Authorization": f"Bearer {openrouter_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "[https://render.com](https://render.com)",
        "X-Title": "AI Location App"
    }
    
    payload = {
        "model": "meta-llama/llama-3.2-11b-vision-instruct:free",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": full_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_data}"}}
                ]
            }
        ],
        "temperature": 0.2
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=payload, timeout=20.0)
        response.raise_for_status()
        res_text = response.json()["choices"][0]["message"]["content"]
        return extract_json_safe(res_text)

async def try_gemini(image_bytes: bytes, mime_type: str, user_hint: str, model_name: str):
    if not gemini_key: raise Exception("Gemini Key is missing")
    client = genai.Client(api_key=gemini_key)
    full_prompt = get_prompt(user_hint)
    config = types.GenerateContentConfig(response_mime_type="application/json", temperature=0.2)
    response = await client.aio.models.generate_content(
        model=model_name,
        contents=[types.Part.from_bytes(data=image_bytes, mime_type=mime_type), full_prompt],
        config=config
    )
    return json.loads(response.text)

async def try_cloudflare(image_bytes: bytes, user_hint: str):
    if not cloudflare_token or not cloudflare_account_id: raise Exception("Cloudflare Credentials missing")
    full_prompt = get_prompt(user_hint)
    
    # 最もペイロードサイズ制限に引っかかりにくい「生バイナリ直接送信」の公式形式
    url = f"[https://api.cloudflare.com/client/v4/accounts/](https://api.cloudflare.com/client/v4/accounts/){cloudflare_account_id}/ai/run/@cf/llava-v1.5-7b-vision-preview"
    headers = {
        "Authorization": f"Bearer {cloudflare_token}",
        "Content-Type": "application/octet-stream"
    }
    
    async with httpx.AsyncClient() as client:
        # プロンプトをURLのクエリパラメータとして逃がし、ボディにバイナリを直撃させる
        response = await client.post(f"{url}?prompt={httpx.URL(full_prompt)}", headers=headers, content=image_bytes, timeout=25.0)
        response.raise_for_status()
        res_text = response.json()["result"]["description"]
        return extract_json_safe(res_text)

@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def read_index(request: Request):
    if request.method == "HEAD": return HTMLResponse(content="", status_code=200)
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...), hint: str = Form(None)):
    image_bytes = await file.read()
    mime_type = file.content_type or "image/jpeg"
    safe_hint = hint if hint else ""

    print("=== 高速フォールバック・ループ開始 ===")
    
    # 1. Groq
    try:
        print("[1番手: Groq] 試行中...")
        data = await try_groq(image_bytes, safe_hint)
        return {"success": True, "reason": data.get("reason") + " (※Groq)", "query_used": data.get("query_used"), "location": data.get("location"), "lat": float(data.get("lat")), "lng": float(data.get("lng"))}
    except Exception as e:
        print(f"→ Groq 失敗: {str(e)[:60]}。OpenRouterへ切り替えます。")

    # 2. OpenRouter
    try:
        print("[2番手: OpenRouter] 試行中...")
        data = await try_openrouter(image_bytes, safe_hint)
        return {"success": True, "reason": data.get("reason") + " (※OpenRouter)", "query_used": data.get("query_used"), "location": data.get("location"), "lat": float(data.get("lat")), "lng": float(data.get("lng"))}
    except Exception as e:
        print(f"→ OpenRouter 失敗: {str(e)[:60]}。Gemini Proへ移行します。")

    # 3. Gemini Pro
    try:
        print("[3番手: Gemini Pro] 試行中...")
        data = await try_gemini(image_bytes, mime_type, safe_hint, 'gemini-2.5-pro')
        return {"success": True, "reason": data.get("reason"), "query_used": data.get("query_used"), "location": data.get("location"), "lat": float(data.get("lat")), "lng": float(data.get("lng"))}
    except Exception as e:
        print(f"→ Gemini Pro 失敗: {str(e)[:60]}。Gemini Flashへ移行します。")

    # 4. Gemini Flash
    try:
        print("[4番手: Gemini Flash] 試行中...")
        data = await try_gemini(image_bytes, mime_type, safe_hint, 'gemini-2.5-flash')
        return {"success": True, "reason": data.get("reason") + " (※Flash)", "query_used": data.get("query_used"), "location": data.get("location"), "lat": float(data.get("lat")), "lng": float(data.get("lng"))}
    except Exception as e:
        print(f"→ Gemini Flash 失敗: {str(e)[:60]}。Cloudflareへ移行します。")

    # 5. Cloudflare Workers AI
    try:
        print("[5番手: Cloudflare] 試行中...")
        data = await try_cloudflare(image_bytes, safe_hint)
        return {"success": True, "reason": data.get("reason") + " (※Cloudflare)", "query_used": data.get("query_used"), "location": data.get("location"), "lat": float(data.get("lat")), "lng": float(data.get("lng"))}
    except Exception as e:
        print(f"→ Cloudflare 失敗: {str(e)[:60]}")

    return {
        "success": False, 
        "message": "すべてのAIへの接続試行がデータ不適合または制限により失敗しました。"
    }
