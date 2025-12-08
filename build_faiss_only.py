import json
from pathlib import Path
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

OUT_DIR = Path("index_out") 
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

chunks_csv = OUT_DIR / "chunks.csv"
meta_json = OUT_DIR / "meta.json"

df = pd.read_csv(chunks_csv)
texts = df["chunk"].tolist()

model = SentenceTransformer(MODEL_NAME)
emb = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

dim = emb.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(emb)

faiss.write_index(index, str(OUT_DIR / "faiss.index"))

meta = df.to_dict(orient="records")
meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

print("done -> faiss.index")
print("chunks:", len(df))
