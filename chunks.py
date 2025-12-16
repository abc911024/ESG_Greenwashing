import re
import json
from pathlib import Path
import fitz
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data")
OUT_DIR = Path("index_out")
OUT_DIR.mkdir(exist_ok=True)

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(MODEL_NAME)

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def norm_company(s: str) -> str:
    """簡單 normalize：去空白、小寫，供比對使用"""
    return re.sub(r"\s+", "", (s or "")).strip().lower()


def load_companies(path: Path = DATA_DIR / "companies.json"):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

def split_sentences(text: str):
    text = normalize_ws(text)
    sents = re.split(r"[。；！？]", text)
    return [s.strip() for s in sents if s.strip()]

def make_chunks(sents, window=3, stride=1):
    chunks = []
    for i in range(0, max(0, len(sents) - window + 1), stride):
        chunk = "。".join(sents[i:i+window]) + "。"
        chunk = normalize_ws(chunk)
        if 50 <= len(chunk) <= 800:
            chunks.append(chunk)
    return chunks

YEAR_4_RE = re.compile(r"(?:19|20)\d{2}")  # 1900~2099

def extract_year_from_text(s: str):
    m = YEAR_4_RE.search(s)
    return int(m.group()) if m else None

def parse_company_year_from_filename(stem: str):
    """
    例：
    - 台灣中油2024 -> company=台灣中油, year=2024
    - TSMC_2023_ESG -> company=TSMC ESG, year=2023
    """
    year = extract_year_from_text(stem)

    company = stem
    if year is not None:
        company = company.replace(str(year), "")

    company = re.sub(r"[_\-]+", " ", company)      # _ - 變空白
    company = re.sub(r"\s+", " ", company).strip() # 多空白收斂

    return company, year

def guess_year_from_pdf_first_page(doc: fitz.Document):
    if len(doc) == 0:
        return None
    t = doc[0].get_text("text", sort=True)
    t = normalize_ws(t)
    return extract_year_from_text(t)

records = []

pdfs = sorted(DATA_DIR.glob("*.pdf"))
if not pdfs:
    raise FileNotFoundError("data/ 資料夾沒有 PDF")

# 載入 canonical companies（由後端/資料提供）
COMPANIES = load_companies()

def canonicalize_company(parsed: str):
    """嘗試把 parsed company 對齊到 companies.json 裡的 canonical value。
    回傳 (canonical_value, company_id)；若找不到，就回傳 (parsed, None)
    """
    if not parsed:
        return parsed, None

    n = norm_company(parsed)
    # 精準比對 -> 包含關係
    for c in COMPANIES:
        v = c.get("value") or c.get("label")
        if not v:
            continue
        nv = norm_company(v)
        if nv == n or nv in n or n in nv:
            return v, c.get("id") or v

    # 找不到就回原始 parsed
    return parsed, None

for pdf_path in pdfs:
    stem = pdf_path.stem
    company, year = parse_company_year_from_filename(stem)

    doc = fitz.open(str(pdf_path))
    if year is None:
        year = guess_year_from_pdf_first_page(doc)

    for pno, page in enumerate(doc, start=1):
        text = page.get_text("text", sort=True)
        text = normalize_ws(text)
        if not text:
            continue

        sents = split_sentences(text)
        chunks = make_chunks(sents, window=3, stride=1)

        for c in chunks:
            canonical, cid = canonicalize_company(company)
            records.append({
                "company": canonical,       # ✅ canonical（來自 companies.json）或 parsed
                "company_id": cid,          # 可為 None
                "year": year,               # ✅ 年分（可能為 None）
                "source_stem": stem,        # ✅ 原始檔名（方便追溯）
                "pdf": str(pdf_path),
                "page": pno,
                "chunk": c
            })

df = pd.DataFrame(records)
df.to_csv(OUT_DIR / "chunks.csv", index=False, encoding="utf-8-sig")

# embeddings
texts = df["chunk"].tolist()
emb = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

dim = emb.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(emb)

faiss.write_index(index, str(OUT_DIR / "faiss.index"))

with open(OUT_DIR / "meta.json", "w", encoding="utf-8") as f:
    json.dump(df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

print("done")
print("chunks:", len(df))
print("index saved to:", OUT_DIR)
