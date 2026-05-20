import asyncio
import os
import urllib.parse
import json
import re
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from google import genai
import httpx

app = FastAPI()

templates = Jinja2Templates(directory="templates")

gemini_key = os.environ.get("GEMINI_API_KEY")

try:
    if gemini_key:
        gemini_client = genai.Client(api_key=gemini_key)
    else:
        gemini_client = None
except Exception as e:
    print(f"Gemini Init Error: {e}")
    gemini_client = None

PROMPT = (
    "Analyze this image and guess the location. "
    "Provide a search query (3-5 words, e.g., 'Eiffel Tower Paris') to find this exact place on a map. "
    "You MUST respond ONLY in the following JSON format. Do not include any markdown block like 
http://googleusercontent.com/immersive_entry_chip/0
http://googleusercontent.com/immersive_entry_chip/1

---

### ✨ 今回追加した「データエラー撲滅」の仕組み

1. **お掃除フィルター（`clean_and_parse_json`）の導入**
   AIが文字の周りに余計な飾り（` ```json ` など）をつけて返してきても、プログラム側で自動的にその飾りを剥ぎ取って、中身のデータだけを綺麗に救出するようにしました。
2. **自動リトライ機能**
   万が一、AIがどうしようもない形式で返答してきた場合は、画面にエラーを出す前に、裏側でもう一度だけ自動でAIに聞き直す（2回目のチャンスをあげる）処理を追加しました。

これでAIの気まぐれによるデータ形式エラーはほとんど発生しなくなります。

保存してRenderの再起動が終わったら、もう一度試してみてください！
