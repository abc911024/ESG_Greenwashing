# agent_d.py
import json
from typing import Dict, Any, List
import ollama

LLM_MODEL = "llama3"  # 或你在用的 qwen2.5:7b 等

def build_claim_brief(agent_a: Dict[str, Any], limit: int = 30) -> List[Dict[str, Any]]:
    """
    把 Agent A 的 claims 縮成適合餵給 LLM 的簡短 JSON
    （保留 claim_text / topic / target_year / certainty / meta_ids）
    """
    claims = agent_a.get("claims", []) or []
    out = []
    for c in claims[:limit]:
        chunks = c.get("source_chunks", []) or []
        out.append({
            "claim_text": c.get("claim_text"),
            "topic": c.get("topic"),
            "target_year": c.get("target_year"),
            "certainty": c.get("certainty"),
            "meta_ids": [ch.get("meta_id") for ch in chunks if ch.get("meta_id") is not None],
        })
    return out

def build_news_brief(agent_c: Dict[str, Any], limit: int = 30) -> List[Dict[str, Any]]:
    """
    把 Agent C 的新聞事件縮成簡短 JSON
    （保留 title / published / summary / link / relevance_score）
    """
    events = agent_c.get("events", []) or []
    out = []
    for e in events[:limit]:
        out.append({
            "title": e.get("title"),
            "published": e.get("published"),
            "summary": e.get("summary"),
            "link": e.get("link"),
            "relevance_score": e.get("relevance_score"),
        })
    return out

def agent_d_judge(query: str, agent_a: Dict[str, Any], agent_c: Dict[str, Any]) -> str:
    """
    Agent D：不再回傳 JSON，而是「給使用者看的文字敘述」。
    這段文字會直接給前端顯示。
    """
    company = agent_a.get("selected_company") or agent_c.get("selected_company") or "目標公司"

    claims_brief = build_claim_brief(agent_a)
    news_brief = build_news_brief(agent_c)

    prompt = f"""
你是「Agent D：漂綠判讀與說明代理人」。
你的任務是：整合永續報告中的承諾（Agent A）以及外部新聞／爭議（Agent C），
用**繁體中文**寫出一份「給一般使用者看的」分析說明，幫助使用者判斷是否有漂綠風險。

【使用者原始問題】
{query}

【推定公司】
{company}

【Agent A：永續報告承諾摘要（JSON，請閱讀後自行整理重點，不要原文貼過來）】
{json.dumps(claims_brief, ensure_ascii=False)}

【Agent C：外部新聞與爭議摘要（JSON，請閱讀後自行整理重點，不要全部貼出）】
{json.dumps(news_brief, ensure_ascii=False)}

請依照下列結構輸出一段「連續文字＋條列」的說明，不要輸出 JSON，不要加任何系統提示：

一、問題與公司重述
- 用 1～3 句話，說明使用者在問什麼、你現在針對哪一家公司在判讀。

二、永續報告中的主要承諾
- 條列 2～6 點「關鍵承諾或政策」，每點說明「主題＋大意」。
- 若能對應到特定 meta_id，請在句尾標註類似「（報告來源：meta_id=822）」的說明即可。

三、外部新聞與爭議重點
- 條列 2～6 則較重要的事件或指控，說明大意與可能影響。
- 若需要引用網址，可簡短標註「（新聞來源）」即可，不用貼完整連結。

四、綜合判讀：漂綠風險評估
- 請明確用一句話標示：「【漂綠風險評估：低】」或「中」或「高」其一。
- 再用 3～6 句話說明你為何這樣判斷：
  - 有沒有「說得很漂亮但外部爭議很多」的 decoupling 現象？
  - 有沒有外部重大事件在報告中幾乎沒被對應（omission）？

五、限制與提醒
- 列出 1～3 點「你無法完全確定的地方」，例如資料時間落差、新聞只是一家媒體觀點等，
  讓使用者知道這不是法律或查核結論，而是一個基於文字證據的輔助判讀。

整篇回答請：
- 一律使用繁體中文。
- 小標題用「一、二、三、四、五、」這種格式。
- 不要輸出 JSON，也不要出現任何類似 {{...}} 或 [] 的結構。
"""

    resp = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": "你是一名嚴謹的永續報告與媒體內容分析專家，會用清楚、有結構的繁體中文寫作。"
            },
            {"role": "user", "content": prompt},
        ],
        options={"temperature": 0.2}
    )

    final_text = resp["message"]["content"].strip()
    return final_text
