# ESG「三角交叉比對」Multi-Agent Demo
![image](https://github.com/abc911024/ESG_Greenwashing/blob/main/%E6%9E%B6%E6%A7%8B%E5%9C%96%20v1.png)
這是一個用 **本地 LLM + 向量資料庫 + Multi-Agent** 做 ESG 漂綠檢核的小型 Demo。  
核心概念是：同時比對企業 **自己在永續報告書中的承諾**，以及 **外部新聞中的負面事件**，再由一個總結 Agent 幫使用者做易懂的文字判讀。

目前包含三個 Agent(AgentB尚在開發中)：

- **Agent A：承諾分析（Commitment Claims）**  
  - 資料來源：企業永續報告書 PDF（向量化後存進 FAISS）
  - 功能：從向量檢索結果中，由本地 LLM 抽取「環境／永續相關承諾」與來源段落
- **Agent C：外部揭露事件（Negative / ESG-related Events）**  
  - 資料來源：Google News RSS 等公開新聞
  - 功能：依公司名稱組合關鍵字，抓出可能相關的負面或 ESG 新聞
- **Agent D：綜合判讀與漂綠風險評估（Greenwashing Assessment）**  
  - 資料來源：Agent A + Agent C 的 JSON 結果
  - 功能：產出給一般使用者閱讀的 **自然語言說明**，討論承諾與實際事件之間是否可能存在落差

前端以一頁式網頁呈現，使用者只要輸入「公司名稱」與「查詢問題」，系統會同時啟動三個 Agents，並在畫面上顯示：

- Agent A：承諾列表與來源段落
- Agent C：新聞候選列表
- Agent D：最後的中文文字判讀（例如：漂綠風險、需要追問的重點）

---

## 專案結構

```bash
ESG_VEC_TEST/
├── app.py                  # FastAPI 主程式（/run API + 靜態頁面掛載）
├── requirements.txt        # 專案所需的套件
├── .gitignore              # 忽略虛擬環境與索引檔等
├── agents/                 # 各個 Agent 的實作
│   ├── __init__.py
│   ├── agent_a.py          # Agent A：ESG 承諾抽取（FAISS + SentenceTransformers + Ollama）
│   ├── agent_c.py          # Agent C：Google News RSS 爬蟲 + 初步格式化
│   └── agent_d.py          # Agent D：整合 A / C 做漂綠風險說明
├── data/                   # 放 ESG 永續報告書 PDF（原始檔）
│   ├── 中油2024.pdf
│   ├── 中石化2024.pdf
│   └── ……
├── index_out/              # 向量索引輸出資料夾（FAISS index + meta.json）
│   ├── faiss.index
│   └── meta.json
├── web/                    # 前端靜態網頁
│   └── index.html          # Tailwind + 原生 JS 單頁介面
├── build_faiss_only.py     # 讀取 data/ PDF → 切 chunk → 建立 FAISS 向量索引（寫入 index_out/）
└── chunks.py               # PDF 解析與 chunk 切分工具
```
## 2. 環境需求

- **Python 3.10+（建議 3.11）**
- **Git**、**VS Code**（或任何你喜歡的 IDE）
- **Ollama**（本地 LLM 服務）
  - 預設模型：`llama3`
  - 也可以改成 `qwen2.5:7b` 等其他支援的本地模型

### 2.1 安裝 Ollama（若尚未安裝）

請到官方網站下載並安裝：

- https://ollama.com/

安裝完成後，在終端機中啟動：

```bash
ollama serve
```
## 3. 安裝相依套件

在啟用 venv 的狀態下執行：

```bash
pip install -r requirements.txt
```
## 4. 建立 ESG 向量索引（一次性步驟）
(這邊的資料是可以變動的 你想處理的永續報告書都可以丟到data內)
在跑 Agent A 之前，需要先把 `data/` 底下的 ESG 永續報告 PDF ：

- 解析成文字
- 切成 chunk
- 編碼成向量並存進 FAISS

專案已經準備好工具腳本：

```bash
python build_faiss_only.py
```
執行成功後，專案根目錄下會自動生成（或更新）`index_out/` 資料夾：
```text
index_out/
├── faiss.index   # FAISS 向量索引檔案
└── meta.json     # 對應的 Metadata（公司名、年度、頁碼、原文內容 chunk）
```
4. 啟動服務
在執行服務前，請再次確認以下事項皆已完成：
[x] 已建立並啟用虛擬環境 (.venv)
[x] 已安裝所有依賴套件 (requirements.txt)
[x] 已成功執行過 build_faiss_only.py 並產出 index_out/
```bash
uvicorn app:app --reload --port 8000
```

接著請打開瀏覽器，前往以下網址即可看到 ESG Multi-Agent Demo 的網頁介面：
http://127.0.0.1:8000


