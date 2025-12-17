# app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
import json

from agents.agent_a import agent_a_extract_claims
from agents.agent_c import agent_c
from agents.agent_d import agent_d_judge  # Agent D：最後的綜合判讀


# ---------- FastAPI 基本設定 ----------
app = FastAPI(title="ESG Greenwashing Multi-Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- 請求 Body 定義 ----------
class RunInput(BaseModel):
    company: str
    query: str


# ---------- 健康檢查 ----------
@app.get("/health")
def health():
    return {"ok": True}


# ---------- 主流程：同時呼叫 A / C / D ----------
@app.post("/run")
def run_all(payload: RunInput):
    """
    前端按下「送出查詢」後會呼叫這裡。

    這裡會依序呼叫：
    - Agent A：永續報告承諾抽取（向量 DB + LLM）
    - Agent C：外部負面新聞爬蟲（Google News RSS）
    - Agent D：綜合判讀，輸出給使用者看的自然語言說明
    """
    company = (payload.company or "").strip()
    query = (payload.query or "").strip()

    # 1) Agent A：查詢時把公司名一起丟進去，提升命中率
    a_query = f"{company} {query}".strip()
    a_result = agent_a_extract_claims(
    query=f"{company} {query}",
    preferred_company=company,
)

    # 2) Agent C：用公司名去抓 Google News RSS
    c_result = agent_c(company)

    # 3) Agent D：產出給使用者看的文字說明（不需再遵守 JSON 格式）
    d_result = agent_d_judge(
        query=f"{company}：{query}".strip("："),
        agent_a=a_result,
        agent_c=c_result,
    )
    # d_result is a dict: {"text":..., "referenced_meta_ids": [...]}
    d_text = d_result.get("text") if isinstance(d_result, dict) else d_result

    return {
        "company": company,
        "query": query,
        "agent_a": a_result,
        "agent_c": c_result,
        "agent_d_text": d_text,
        "agent_d": d_result,
    }


@app.get("/companies")
def get_companies():
    """回傳後端 canonical companies 清單，供前端選單使用。"""
    p = Path("data") / "companies.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []


# ---------- 靜態檔案（前端頁面） ----------
# 一定要放在「所有 route 宣告之後」，避免吃掉 /run, /health
app.mount("/", StaticFiles(directory="web", html=True), name="web")
