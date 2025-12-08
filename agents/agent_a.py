import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

import faiss
from sentence_transformers import SentenceTransformer
import ollama


# ===== 路徑 =====
OUT_DIR = Path("index_out")
INDEX_PATH = OUT_DIR / "faiss.index"
META_PATH = OUT_DIR / "meta.json"

# ===== 模型 =====
EMB_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "llama3"  # 若你 Ollama 沒有 llama3，可改 qwen2.5:7b / qwen2.5:3b

# ===== 控制參數 =====
RETRIEVE_TOPK = 500
COMPANY_TOPN = 5               # 做候選排序（最後只取 Top1）
PASSAGES_FOR_SELECTED = 30     # 被選中公司要給 LLM 的片段數
CITE_CHUNK_MAXLEN = 160        # 最終輸出 source_chunks 時的節錄長度


# ===== 讀取 index / meta =====
index = faiss.read_index(str(INDEX_PATH))
meta = json.loads(META_PATH.read_text(encoding="utf-8"))

embedder = SentenceTransformer(EMB_MODEL)


# ===== 小工具 =====
def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def truncate(s: str, max_len: int) -> str:
    s = normalize_ws(s)
    if max_len and len(s) > max_len:
        return s[:max_len] + "..."
    return s


def dedupe_claims(claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    依照 (company, claim_text, topic, metric) 去重，
    同一承諾若出現多次，就合併 citations & source_chunks。
    """
    from copy import deepcopy

    seen = {}
    result = []

    for c in claims:
        key = (
            c.get("company", ""),
            normalize_ws(c.get("claim_text", "")),
            c.get("topic", ""),
            c.get("metric", ""),
        )
        if key in seen:
            existing = seen[key]
            # 合併 source_citations
            cites_old = [str(x) for x in existing.get("source_citations", [])]
            cites_new = [str(x) for x in c.get("source_citations", [])]
            merged_cites = sorted(set(cites_old + cites_new))
            existing["source_citations"] = merged_cites

            # 合併 source_chunks（簡單 append）
            chunks_old = existing.get("source_chunks", [])
            chunks_new = c.get("source_chunks", []) or []
            existing["source_chunks"] = chunks_old + chunks_new
        else:
            clone = deepcopy(c)
            seen[key] = clone
            result.append(clone)

    return result


def retrieve_all(query: str, topk: int = 50) -> List[Dict[str, Any]]:
    """
    從 FAISS 取回 topk
    新增 meta_id（= i）保留原始 chunk 身分
    """
    q_emb = embedder.encode([query], normalize_embeddings=True)
    scores, ids = index.search(q_emb, topk)

    results = []
    for i, s in zip(ids[0], scores[0]):
        if i < 0:
            continue
        m = meta[i]
        results.append(
            {
                "meta_id": int(i),  # ✅ 關鍵：保留 meta 原始編號
                "score": float(s),
                "company": m.get("company", ""),
                "year": m.get("year", ""),
                "page": m.get("page", ""),
                "chunk": normalize_ws(m.get("chunk", "")),
            }
        )
    return results


def rank_companies(results: List[Dict[str, Any]], topn: int = 3) -> List[Tuple[str, float]]:
    agg = defaultdict(float)
    for r in results:
        c = r.get("company")
        if c:
            agg[c] += r["score"]

    ranked = sorted(agg.items(), key=lambda x: x[1], reverse=True)
    return ranked[:topn]


def pick_company_passages(
    results: List[Dict[str, Any]], company: str, limit: int = 5
) -> List[Dict[str, Any]]:
    cands = [r for r in results if r.get("company") == company]
    cands.sort(key=lambda x: x["score"], reverse=True)
    return cands[:limit]


def build_context(passages: List[Dict[str, Any]]) -> Tuple[str, Dict[int, Dict[str, Any]]]:
    """
    使用 meta_id 當引用編號
    回傳：
      - context_str
      - cite_map: meta_id -> passage
    """
    lines = []
    cite_map: Dict[int, Dict[str, Any]] = {}

    for p in passages:
        mid = int(p.get("meta_id"))
        lines.append(
            f"[{mid}] 公司:{p.get('company','')} | 年度:{p.get('year','')} | 頁碼:{p.get('page','')} | 內容:{p.get('chunk','')}"
        )
        cite_map[mid] = p

    return "\n".join(lines), cite_map


def safe_parse_json(text: str):
    text = (text or "").strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\[[\s\S]*\]", text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None

    return None


def enrich_claims_with_source_chunks(
    claims: List[Dict[str, Any]],
    cite_map: Dict[int, Dict[str, Any]],
    chunk_maxlen: int = 160,
) -> List[Dict[str, Any]]:
    """
    source_citations 現在會是 meta_id
    所以直接用 meta_id 去 cite_map 找回 chunk
    """
    for c in claims:
        cites = c.get("source_citations", [])
        source_chunks = []

        for n in cites:
            try:
                mid = int(n)
            except Exception:
                continue

            p = cite_map.get(mid)
            if not p:
                continue

            source_chunks.append(
                {
                    "meta_id": mid,
                    "company": p.get("company", ""),
                    "year": p.get("year", ""),
                    "page": p.get("page", ""),
                    "score": p.get("score", None),
                    "chunk": truncate(p.get("chunk", ""), chunk_maxlen),
                }
            )

        c["source_chunks"] = source_chunks

    return claims


# ===== 公司名稱比對 + 選擇 =====
def match_company_name(name: str, preferred: str) -> bool:
    """判斷公司名稱是否與使用者輸入相符（用包含關係 + 去掉空白）"""
    def norm(s: str) -> str:
        return re.sub(r"\s+", "", s or "").lower()

    n1 = norm(name)
    n2 = norm(preferred)
    if not n1 or not n2:
        return False
    # 例如：「台塑」 vs 「台塑2024」
    return n2 in n1 or n1 in n2


def choose_company(
    company_ranked: List[Tuple[str, float]],
    preferred_company: Optional[str] = None,
) -> Optional[str]:
    """
    優先選「名字裡有使用者輸入公司」的那一家。
    若有指定公司但沒有任何 match，就回傳 None。
    若沒指定公司，才 fallback 到最高分的公司。
    """
    if not company_ranked:
        return None

    if preferred_company:
        for name, _score in company_ranked:
            if match_company_name(name, preferred_company):
                return name

        # 有指定公司，但在排名中找不到任何 match，就不要亂選別的公司
        return None

    # 沒指定公司時，維持原本行為：用 top1
    return company_ranked[0][0]


# ===== LLM：抽承諾 =====
def ask_llm_extract_claims(selected_company: str, passages: List[Dict[str, Any]]):
    context, cite_map = build_context(passages)

    prompt = f"""
【強制輸出格式】
你只能輸出「JSON 陣列」且不得包含任何其他文字。
若無明確承諾，請輸出 []。

你是永續報告書的「承諾提取代理人（Agent A）」。
你的任務是：只根據引用內容，抽取企業在環境/永續面向的承諾、目標、策略、路徑或政策宣示。

【本次判斷的公司】
{selected_company}

【引用內容（僅可使用以下內容）】
{context}

【輸出要求】
1) 僅能根據引用內容，不可自行推測或補寫。
2) 請用**繁體中文**。
3) 請以「JSON 陣列」輸出。
4) 每筆物件包含下列欄位：
   - company（請填 {selected_company}）
   - claim_text（請盡量抽出「完整承諾句」，避免只有章節標題）
   - topic（climate/water/waste/energy/biodiversity/general）
   - target_year（沒有就填 null）
   - metric（例如 GHG/renewable/energy_efficiency/SS/COD/Oil/Phenol/unknown）
   - certainty（high/medium/low）
   - source_citations（請使用「引用內容中的編號」，也就是方括號內的 meta 原始編號，例如 [123]）
5) 若引用內容沒有明確承諾，輸出 []。
6) 你可以輸出超過20筆，只要引用內容支持即可。
7) 若同一承諾出現多次，請合併為一筆。
請只輸出 JSON。
"""

    # 第一次（強約束 + 低溫）
    resp = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a strict JSON generator. Output ONLY a JSON array. No explanations.",
            },
            {"role": "user", "content": prompt},
        ],
        options={"temperature": 0},
    )
    raw = resp["message"]["content"].strip()

    if safe_parse_json(raw) is not None:
        return raw, cite_map

    # 第二次：修復 JSON
    repair_prompt = f"""
你剛才的輸出不是合法 JSON。
請根據下列引用內容，重新輸出「只包含 JSON 陣列」的答案。

【本次判斷的公司】
{selected_company}

【引用內容】
{context}

【輸出欄位】
- company
- claim_text
- topic
- target_year
- metric
- certainty
- source_citations（使用方括號內的 meta 原始編號）

只能輸出 JSON 陣列。若無明確承諾，輸出 []。
"""

    resp2 = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a strict JSON generator. Output ONLY a JSON array. No explanations.",
            },
            {"role": "user", "content": repair_prompt},
        ],
        options={"temperature": 0},
    )
    raw2 = resp2["message"]["content"].strip()
    return raw2, cite_map


# ===== Agent A 主流程（跨公司自動判斷）=====
def agent_a_extract_claims(query: str, preferred_company: Optional[str] = None):
    """
    query：檢索用的文字（建議已經把公司名一起塞進來）
    preferred_company：使用者在前端輸入的「公司名稱」，用來優先選對公司
    """
    print("[AgentA] 收到 query =", query, "| preferred_company =", preferred_company)

    # 1) 全庫檢索
    print("[AgentA] 開始 retrieve_all ...")
    results = retrieve_all(query, topk=RETRIEVE_TOPK)
    print(f"[AgentA] retrieve_all 完成，取回 {len(results)} 筆")

    if not results:
        return {"ok": True, "selected_company": preferred_company, "claims": []}

    # === 2) 如果有指定公司，就先過濾掉別的公司 ===
    if preferred_company:
        filtered = [
            r
            for r in results
            if match_company_name(r.get("company", ""), preferred_company)
        ]
        print(f"[AgentA] 針對 {preferred_company} 過濾後剩 {len(filtered)} 筆")

        # 完全沒有命中 -> 回傳「這家公司，但沒有找到承諾」
        if not filtered:
            return {
                "ok": True,
                "selected_company": preferred_company,
                "claims": [],
            }

        # 用過濾後的結果排公司分數（通常只有一個，例如「台塑2024」）
        company_ranked = rank_companies(filtered, topn=COMPANY_TOPN)
        print("[AgentA] company_ranked (filtered) =", company_ranked)

        selected_company = choose_company(company_ranked, preferred_company)
        print("[AgentA] 最後選中的公司 =", selected_company)

        if not selected_company:
            return {
                "ok": True,
                "selected_company": preferred_company,
                "claims": [],
            }

        selected_passages = pick_company_passages(
            filtered, selected_company, limit=PASSAGES_FOR_SELECTED
        )
    else:
        # === 沒指定公司 → 保留原本「自動判斷哪家公司」的邏輯 ===
        company_ranked = rank_companies(results, topn=COMPANY_TOPN)
        print("[AgentA] company_ranked =", company_ranked)
        if not company_ranked:
            return {"ok": True, "selected_company": None, "claims": []}

        selected_company = choose_company(company_ranked, None)
        print("[AgentA] 最後選中的公司 =", selected_company)
        if not selected_company:
            return {"ok": True, "selected_company": None, "claims": []}

        selected_passages = pick_company_passages(
            results, selected_company, limit=PASSAGES_FOR_SELECTED
        )

    print(f"[AgentA] 對 {selected_company} 取出 {len(selected_passages)} 個 passages")

    # 5) 丟給本地 LLM 抽承諾清單
    print("[AgentA] 準備呼叫 Ollama LLM ...")
    raw, cite_map = ask_llm_extract_claims(selected_company, selected_passages)
    print("[AgentA] Ollama 回來了，長度 =", len(raw))

    parsed = safe_parse_json(raw)

    # 6) 若 LLM 沒吐合法 JSON，就保底回傳 raw
    if parsed is None:
        return {
            "ok": False,
            "selected_company": selected_company,
            "raw": raw,
            "claims": [],
        }

    # 7) 後處理：把 citations（meta_id）轉成 chunks，並去重
    parsed = enrich_claims_with_source_chunks(
        parsed, cite_map, chunk_maxlen=CITE_CHUNK_MAXLEN
    )
    parsed = dedupe_claims(parsed)

    return {
        "ok": True,
        "selected_company": selected_company,
        "claims": parsed,
    }


# ===== 直接執行檔案時（CLI 測試用）=====
if __name__ == "__main__":
    import sys

    # 支援：python agents/agent_a.py 台塑 請問在永續報告書內有提到哪些罰款？
    if len(sys.argv) > 2:
        preferred = sys.argv[1]
        user_query = " ".join(sys.argv[2:])
    else:
        preferred = None
        user_query = input(
            "請輸入查詢問題（例如：請問台灣中油在永續報告書內有提到那些罰款？）："
        ).strip()

    result = agent_a_extract_claims(user_query, preferred_company=preferred)

    out_path = Path("agentA_claims.json")

    if result["ok"]:
        payload = {
            "selected_company": result.get("selected_company"),
            "claims": result.get("claims", []),
        }
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        # 保底：LLM 輸出不是 JSON
        out_path.write_text(result.get("raw", ""), encoding="utf-8")
        print(result.get("raw", ""))
