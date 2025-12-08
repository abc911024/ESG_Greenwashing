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

records = []

pdfs = sorted(DATA_DIR.glob("*.pdf"))
if not pdfs:
    raise FileNotFoundError("data/ 資料夾沒有 PDF")

for pdf_path in pdfs:
    doc = fitz.open(str(pdf_path))
    company = pdf_path.stem

    for pno, page in enumerate(doc, start=1):
        text = page.get_text("text", sort=True)
        text = normalize_ws(text)
        if not text:
            continue

        sents = split_sentences(text)
        chunks = make_chunks(sents, window=3, stride=1)

        for c in chunks:
            records.append({
                "company": company,
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
index = faiss.IndexFlatIP(dim)  # 內積（搭配 normalize 就是 cosine）
index.add(emb)

faiss.write_index(index, str(OUT_DIR / "faiss.index"))

# 存一份 row 對應表
with open(OUT_DIR / "meta.json", "w", encoding="utf-8") as f:
    json.dump(df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

print("done")
print("chunks:", len(df))
print("index saved to:", OUT_DIR)
