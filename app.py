import json
import os
import re
from pathlib import Path
from urllib.parse import quote

import httpx
from dotenv import load_dotenv  # .env 읽는 도구
from flask import Flask, jsonify, render_template, request
from groq import BadRequestError, Groq, RateLimitError

# 1. app.py 옆의 .env 키들을 환경변수로 불러옴
# (경로를 안 주면 VS Code 디버거 실행 시 작업 폴더의 다른 .env를 읽을 수 있음)
load_dotenv(Path(__file__).with_name(".env"))

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
MAX_HISTORY = 20  # 서버로 받아들이는 최근 대화 개수
MAX_PICKS = 3  # 한 번에 추천하는 가게 수

# 토큰 절약을 위해 AI 지시문은 영어로 쓰고, 답변은 한국어로 하게 한다
BASE_PROMPT = (
    "You are '부천이', a friendly local guide for Bucheon, Korea. "
    "Help with restaurants, cafes, outings and transport in Bucheon. Always reply in Korean. "
    "Never invent opening hours or prices; tell the user to check before visiting. "
    "Never recommend places outside Bucheon-si (원미구, 소사구, 오정구). "
    "For unrelated questions, answer briefly and steer back to Bucheon."
)

OTHER_AREA_ANSWER = (
    "저는 부천시(원미구·소사구·오정구) 안의 장소만 안내해요. "
    "부천에서 찾고 싶은 음식이나 장소를 알려주시면 찾아드릴게요! 예: '부천역 삼겹살', '중동 카페'"
)

NOT_FOUND_ANSWER = (
    "부천에서 '{query}'에 맞는 곳을 네이버 지도에서 찾지 못했어요. "
    "'부천역 삼겹살', '상동 카페'처럼 지역과 음식(장소) 종류로 물어봐 주세요."
)

# 네이버 키가 없을 때만 쓰는 방식 (AI가 상호명을 직접 씀 → 지어낼 수 있음)
NAME_PROMPT = (
    "\n\nWhen recommending, use exact business names (not categories), wrapped like [[상호명]]. "
    "At most 3 places. Do not recommend any place you are not sure exists."
)

# 네이버 목록이 있을 때: AI는 번호만 고르고, 상호명은 서버가 채운다
PICK_PROMPT = (
    "\n\n[Real places from Naver Map]\n{places}\n\n"
    "Pick up to {max_picks} places from this list that fit the user's question. Reply ONLY with JSON:\n"
    '{{"intro": "...", "picks": [{{"id": <number>, "reason": "..."}}], "outro": "..."}}\n'
    "Rules:\n"
    "- Pick by list number only. Never write any place name in intro/reason/outro; names are filled in automatically.\n"
    "- reason: one short line on why it fits the question (e.g. '부천역에서 가까운 초밥집'). "
    "Don't repeat the category or address; they are shown separately.\n"
    "- Never invent info not in the list (menu, floor, mood, hours).\n"
    "- If nothing fits, use an empty picks array and say so in intro.\n"
    "- Never mention places not in this list, even if they came up earlier.\n"
    "- Write intro, reason and outro in Korean."
)

NO_NAMES_PROMPT = (
    "\n\nDo not mention any business or place names in this reply, including ones from earlier in the chat. "
    "If the user is looking for a place, ask what food or area they want."
)

QUERY_PROMPT = (
    "If the user's last message is looking for a place in Bucheon (restaurant, cafe, activity, "
    "sightseeing, kid-friendly spot, etc.), output ONE short Korean Naver Map search query: area + type, "
    "2-3 words. Examples: '부천역 삼겹살', '중동 파스타', '송내 카페', '부천 아이와 가볼만한곳', '부천 공원'. "
    "Prefix '부천' if no area is given. "
    "These are all inside Bucheon: 중동, 신중동, 상동, 심곡, 원미, 춘의, 도당, 약대, 소사, 소사본동, 송내, "
    "괴안, 범박, 옥길, 역곡, 오정, 원종, 고강, 신흥, 작동, 여월, 성곡, 까치울, and stations 부천역, 송내역, "
    "역곡역, 소사역, 중동역, 신중동역, 부천시청역, 상동역, 춘의역, 까치울역, 부천종합운동장역. "
    "If it asks for a place in another area (인천, 부평, 계양, 서울, 시흥, 광명, 김포, etc.), output only OTHER. "
    "If it is not looking for a place (greeting, chat, transport), output only NONE. No other text."
)

# 2. 키로 Groq 클라이언트 만들기
groq_key = os.getenv("GROQ_API_KEY")
gemini_key = os.getenv("GEMINI_API_KEY")
if gemini_key and not gemini_key.isascii():
    gemini_key = None  # '여기에_gemini_키' 같은 자리표시자는 키가 없는 것으로 본다
naver_id = os.getenv("NAVER_CLIENT_ID")
naver_secret = os.getenv("NAVER_CLIENT_SECRET")
use_naver = bool(naver_id and naver_secret)
client = Groq(api_key=groq_key) if groq_key else None

app = Flask(__name__)


def ask_gemini(messages, json_mode=False):
    """Groq 한도 소진 시 사용하는 Gemini 호출 (OpenAI 호환 엔드포인트)."""
    if not gemini_key:
        raise RuntimeError("Groq 사용 한도를 초과했고 GEMINI_API_KEY도 없습니다. 잠시 후 다시 시도해 주세요.")
    body = {"model": GEMINI_MODEL, "messages": messages}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    r = httpx.post(
        GEMINI_URL,
        headers={"Authorization": f"Bearer {gemini_key}"},
        json=body,
        timeout=20,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def ask(messages, json_mode=False):
    if client:
        try:
            extra = {"response_format": {"type": "json_object"}} if json_mode else {}
            res = client.chat.completions.create(model=GROQ_MODEL, messages=messages, **extra)
            return res.choices[0].message.content
        except RateLimitError:
            pass  # Gemini로 전환
    return ask_gemini(messages, json_mode)


def make_search_query(history):
    """대화에서 네이버 지도 검색어를 뽑는다. 장소 질문이 아니면 None, 부천 밖 지역이면 "OTHER"."""
    recent = history[-4:]
    text = ask([{"role": "system", "content": QUERY_PROMPT}] + recent)
    query = (text or "").strip().splitlines()[0].strip().strip("'\"") if text and text.strip() else ""
    if not query or query.upper() == "NONE":
        return None
    if query.upper() == "OTHER":
        return "OTHER"
    return query if "부천" in query else "부천 " + query


def search_places(query):
    """네이버 지역 검색 API로 실제 가게를 조회한다. 실패하면 빈 목록."""
    if not use_naver:
        return []
    try:
        r = httpx.get(
            "https://openapi.naver.com/v1/search/local.json",
            headers={"X-Naver-Client-Id": naver_id, "X-Naver-Client-Secret": naver_secret},
            params={"query": query, "display": 5, "sort": "comment"},
            timeout=10,
        )
        r.raise_for_status()
    except httpx.HTTPError:
        return []

    places = []
    for it in r.json().get("items", []):
        name = re.sub(r"<[^>]+>", "", it["title"]).strip()
        addr = it.get("roadAddress") or it.get("address") or ""
        # '부천로', '부천빌딩'처럼 다른 지역 주소에 든 '부천'은 거르고, 부천시 주소만 남긴다
        if not re.match(r"경기(도)?\s+부천시\s", addr):
            continue
        # 같은 이름의 다른 지점과 구분되도록 시/구까지 넣어 검색한다
        area = " ".join(addr.split()[1:3])
        places.append({
            "name": name,
            "category": it.get("category", ""),
            "address": addr,
            "url": "https://map.naver.com/p/search/" + quote(f"{name} {area}".strip()),
        })
    return places


def strip_brackets(text):
    """[[ ]]로 감싼 이름이 든 줄을 지운다. 목록 없이 나온 이름은 전부 지어낸 것이기 때문."""
    lines = [ln for ln in text.splitlines() if not re.search(r"\[\[.+?\]\]", ln)]
    return "\n".join(lines).strip()


def answer_with_places(clean, places):
    """AI에게 번호만 고르게 하고, 상호명·주소는 네이버 데이터로 채워서 답변을 만든다."""
    listing = "\n".join(
        f"{i}. {p['name']} | {p['category']} | {p['address']}" for i, p in enumerate(places, 1)
    )
    system = BASE_PROMPT + PICK_PROMPT.format(places=listing, max_picks=MAX_PICKS)
    try:
        data = json.loads(ask([{"role": "system", "content": system}] + clean, json_mode=True))
    except (BadRequestError, httpx.HTTPStatusError, TypeError, json.JSONDecodeError):
        data = {}  # AI가 JSON 형식을 어기면 아래에서 네이버 결과를 그대로 보여준다

    picks, seen = [], set()
    for pick in data.get("picks") or []:
        try:
            idx = int(pick.get("id")) - 1
        except (AttributeError, TypeError, ValueError):
            continue
        if 0 <= idx < len(places) and idx not in seen:
            seen.add(idx)
            picks.append((places[idx], strip_brackets(str(pick.get("reason", "")))))
    picks = picks[:MAX_PICKS]

    intro = strip_brackets(str(data.get("intro", "")))
    outro = strip_brackets(str(data.get("outro", "")))
    if not data:  # JSON을 못 받았으면 네이버 결과를 그대로 보여준다
        intro = "네이버 지도에서 찾은 곳들이에요."
        picks = [(p, "") for p in places[:MAX_PICKS]]

    parts = [intro] if intro else []
    for p, reason in picks:
        line = f"- [[{p['name']}]] ({p['category']}, {p['address']})"
        parts.append(line + (f"\n  {reason}" if reason else ""))
    if picks:
        parts.append(outro or "영업시간과 가격은 방문 전에 확인해 주세요.")
    elif not intro:
        parts.append("질문에 맞는 가게를 찾지 못했어요. 음식 종류나 지역을 바꿔서 다시 물어봐 주세요.")
    elif outro:
        parts.append(outro)
    return "\n\n".join(parts)


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/chat")
def chat():
    history = (request.get_json(silent=True) or {}).get("messages", [])
    # 클라이언트가 보낸 대화 기록 중 user/assistant 문자열만 받아들임
    clean = [
        {"role": m["role"], "content": m["content"]}
        for m in history
        if isinstance(m, dict)
        and m.get("role") in ("user", "assistant")
        and isinstance(m.get("content"), str)
    ][-MAX_HISTORY:]
    if not clean or clean[-1]["role"] != "user":
        return jsonify(error="질문이 비어 있습니다."), 400

    try:
        places = []
        if not use_naver:
            answer = ask([{"role": "system", "content": BASE_PROMPT + NAME_PROMPT}] + clean)
        else:
            query = make_search_query(clean)
            if query == "OTHER":
                answer = OTHER_AREA_ANSWER
            elif query:
                places = search_places(query)
                # 검색 결과가 없으면 AI에게 맡기지 않는다 (맡기면 가게를 지어냄)
                answer = answer_with_places(clean, places) if places else NOT_FOUND_ANSWER.format(query=query)
            else:
                answer = strip_brackets(ask([{"role": "system", "content": BASE_PROMPT + NO_NAMES_PROMPT}] + clean))
    except Exception as e:
        return jsonify(error=f"답변을 가져오지 못했습니다: {e}"), 502

    # 화면에서 [[상호명]]을 정확한 지도 링크로 바꾸는 데 쓴다
    return jsonify(answer=answer, links={p["name"]: p["url"] for p in places})


if __name__ == "__main__":
    if not groq_key and not gemini_key:
        raise SystemExit(".env에 GROQ_API_KEY(또는 GEMINI_API_KEY)를 설정해주세요.")
    if not use_naver:
        print("⚠️ NAVER_CLIENT_ID/SECRET이 없어 네이버 조회 없이 동작합니다. 없는 가게가 추천될 수 있습니다.")
    app.run(debug=True)
