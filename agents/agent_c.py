import json
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from urllib.parse import urlencode
import feedparser
from sentence_transformers import SentenceTransformer
import ollama


# ===== 模型 =====
EMB_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "llama3"  # 或你本機有的模型

embedder = SentenceTransformer(EMB_MODEL)

# ===== 參數 =====
TOPK_FEEDS_PER_QUERY = 50
EMB_FILTER_TOPK = 12

# 你可自己擴充媒體 RSS
GOOGLE_NEWS_RSS = "https://news.google.com/{query}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"


# ===== 小工具 =====
def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def build_queries(company: str) -> List[str]:
    base = company.strip()
    return [
        f"{base} 永續 罰款",
        f"{base} 污染",
        f"{base} 環保 裁罰",
        f"{base} 漏油",
        f"{base} 火災 爆炸",
        f"{base} 碳排 放空",
    ]


def fetch_google_rss(query: str, limit: int = 10):
    base = "https://news.google.com/rss/search"

    params = {
        "q": query,          # 讓 urlencode 處理空白與中文
        "hl": "zh-TW",
        "gl": "TW",
        "ceid": "TW:zh-Hant"
    }

    url = f"{base}?{urlencode(params)}"

    feed = feedparser.parse(url)

    items = []
    for e in feed.entries[:limit]:
        items.append({
            "title": getattr(e, "title", ""),
            "link": getattr(e, "link", ""),
            "published": getattr(e, "published", ""),
            "summary": getattr(e, "summary", ""),
            "source": "google_rss",
            "query": query,
        })
    return items


def parse_date_safe(published_str: str) -> Optional[str]:
    if not published_str:
        return None
    # Google RSS 常見格式可直接嘗試解析
    try:
        dt = datetime(*feedparser._parse_date(published_str)[:6])
        return dt.strftime("%Y-%m-%d")
    except:
        return None


def dedup_news(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for it in items:
        key = (it.get("title", ""), it.get("url", ""))
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def emb_rerank(company: str, items: List[Dict[str, Any]], topk: int = 10) -> List[Dict[str, Any]]:
    if not items:
        return []

    query = f"{company} 環境 永續 負面事件 裁罰 污染"
    texts = [f"{it.get('title','')} {it.get('summary','')}" for it in items]

    q_emb = embedder.encode([query], normalize_embeddings=True)
    d_emb = embedder.encode(texts, normalize_embeddings=True)

    # cosine via dot
    scores = (d_emb @ q_emb[0]).tolist()

    ranked = sorted(zip(items, scores), key=lambda x: x[1], reverse=True)[:topk]
    out = []
    for it, s in ranked:
        it = dict(it)
        it["relevance_score"] = float(s)
        it["event_date_guess"] = parse_date_safe(it.get("published", "")) or None
        out.append(it)
    return out


def safe_parse_json(text: str):
    text = (text or "").strip()
    try:
        return json.loads(text)
    except:
        pass

    m = re.search(r"\{[\s\S]*\}$", text)
    if m:
        try:
            return json.loads(m.group(0))
        except:
            return None
    return None


def ask_llm_extract_events(company: str, news_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    # 給 LLM 的引用上下文（用編號）
    lines = []
    for i, it in enumerate(news_items, start=1):
        lines.append(
            f"[{i}] 標題:{it.get('title','')} | 日期:{it.get('event_date_guess') or ''} | 摘要:{it.get('summary','')} | URL:{it.get('url','')}"
        )
    context = "\n".join(lines)

    prompt = f"""
【強制輸出格式】
你只能輸出「JSON 物件」且不得包含任何其他文字。

你是「輿情揭露代理人（Agent C）」。
任務：根據新聞引用內容，抽取與 {company} 相關的「環境/永續負面事件」。
若引用內容不足以形成負面事件，events 請輸出空陣列。

【公司】
{company}

【新聞引用】
{context}

【輸出 JSON 格式】
{{
  "selected_company": "{company}",
  "events": [
    {{
      "event_id": "news_0001",
      "company": "{company}",
      "event_title": "...",
      "event_text": "用繁體中文描述事件重點（2-4句）",
      "event_date": "YYYY-MM-DD 或 null",
      "topic": "climate/water/waste/energy/general",
      "severity": "high/medium/low",
      "source_citations": [1,2],
      "evidence": {{
        "snippet": "從引用中挑最關鍵一句或一小段（繁中）"
      }}
    }}
  ]
}}

規則：
1) 只能根據引用內容，不可自行推測。
2) 只能引用出現在新聞引用中的編號。
3) topic/severity 請保守判斷。
"""

    resp = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are a strict JSON generator. Output ONLY JSON."},
            {"role": "user", "content": prompt}
        ],
        options={"temperature": 0}
    )

    raw = resp["message"]["content"]
    parsed = safe_parse_json(raw)
    if parsed is not None:
        return parsed

    # fallback
    return {"selected_company": company, "events": []}


def enrich_event_sources(events_payload: Dict[str, Any], news_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    idx_map = {i+1: it for i, it in enumerate(news_items)}

    for ev in events_payload.get("events", []):
        cites = ev.get("source_citations", [])
        sources = []
        for c in cites:
            try:
                ci = int(c)
            except:
                continue
            it = idx_map.get(ci)
            if not it:
                continue
            sources.append({
                "title": it.get("title"),
                "url": it.get("url"),
                "published": it.get("published"),
                "summary": it.get("summary"),
                "relevance_score": it.get("relevance_score"),
            })
        ev["sources"] = sources

    return events_payload


def agent_c(company: str) -> Dict[str, Any]:
    # 1) build queries
    queries = build_queries(company)

    # 2) fetch rss
    all_items = []
    for q in queries:
        all_items.extend(fetch_google_rss(q, limit=TOPK_FEEDS_PER_QUERY))

    all_items = dedup_news(all_items)

    # 3) emb rerank
    top_news = emb_rerank(company, all_items, topk=EMB_FILTER_TOPK)

    # 4) llm extract
    payload = ask_llm_extract_events(company, top_news)

    # 5) attach sources
    payload = enrich_event_sources(payload, top_news)

    # 6) add query trace
    payload["query"] = queries
    payload["candidates_used"] = top_news

    return payload


if __name__ == "__main__":
    company = "台灣中油"
    result = agent_c(company)

    out_path = Path("agentC_events.json")
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
