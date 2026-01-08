# Taipei Art Exhibition Data Pipeline (GenAI & OCR Powered)

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Google Gemini](https://img.shields.io/badge/GenAI-Google_Gemini-8E75B2?style=for-the-badge&logo=google&logoColor=white)
![EasyOCR](https://img.shields.io/badge/OCR-EasyOCR-FFD43B?style=for-the-badge)
![Selenium](https://img.shields.io/badge/Web_Scraping-Selenium-43B02A?style=for-the-badge&logo=selenium&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/Database-Supabase-3ECF8E?style=for-the-badge&logo=supabase&logoColor=white)

> 這是一個針對台北市各大藝文場館（如故宮、北美館、松菸等）的自動化數據工程專案。
> 專案核心解決了 **非結構化數據 (Unstructured Data)** 的自動化提取難題，整合了 **OCR (光學文字辨識)** 與 **LLM (Google Gemini)** 技術，將複雜的> 網頁佈局與圖片資訊轉化為標準化的資料庫格式。

---

## Architecture (系統架構)

> 本專案採用 **模組化設計 (Modular Design)** 與 **策略模式 (Strategy Pattern)**，確保對不同場館的擴充性。

```mermaid
graph TD
    %% ===== Extract Layer =====
    subgraph "Extract Layer"
        A["Job Controller"] -->|Dispatch| B{"Crawler Router"}
        B -->|Static HTML| C["HTTP Crawler (Requests + BS4)"]
        B -->|JS Rendered| D["Browser Crawler (Selenium)"]
        B -->|Image Based| E["OCR Pipeline (OpenCV + EasyOCR)"]
    end

    %% ===== Transform Layer =====
    subgraph "Transform Layer"
        C -->|Unstructured Text| X["Unified Raw Input"]
        D -->|Unstructured Text| X
        E -->|Image to Text| X

        X -->|Context + Prompt| F["LLM Parser (Gemini API)"]
        F -->|Structured JSON| G["Schema Validation & Normalization"]
        G -->|Address Field| H["Geocoding Service"]
    end

    %% ===== Load & Serve Layer =====
    subgraph "Load & Serving Layer"
        G -->|Upsert| I["Primary Database (PostgreSQL)"]
        I -->|Read API| J["Data Access Layer"]
        J -->|Visualization| K["Streamlit Dashboard"]
    end

    %% ===== Styles =====
    style F fill:#8E75B2,stroke:#333,stroke-width:2px,color:white
    style E fill:#FFD43B,stroke:#333,stroke-width:2px,color:black
    style K fill:#FF4B4B,stroke:#333,stroke-width:2px,color:white

```

## Key Features (核心技術亮點)
1. Hybrid Scraping Strategy (混合爬取策略)
    針對不同網站特性採用最佳化方案：

    - Static: 使用 Requests + BeautifulSoup 處理結構簡單網站 (如：師大美術館)。
    - Dynamic: 使用 Selenium 處理動態載入與分頁互動 (如：北美館)。
    - Visual: 使用 EasyOCR + OpenCV 針對「圖片形式的票價表」進行文字提取 (如：富邦美術館、松山文創園區)。

2. GenAI-Powered Transformation (AI 驅動轉換)
    解決傳統 Regex 無法處理的語意提取問題：
    使用 Google Gemini 2.0 Flash 模型進行非結構化文本的 JSON 實體抽取。
    實作 Retry Mechanism 與 Self-Correction Prompt，當 AI 產出格式錯誤時，自動要求模型修正。

3. Engineering Best Practices (工程實踐)
    OOP Design: 定義 ExhibitionETLPipeline 介面與 exhibition_data dataclass，確保資料一致性。
    Error Handling: 針對 Network, API Rate Limit, Parsing Error 建立完整的 try-except 與 Log 機制。
    Config Management: 使用 .env 管理 API Keys 與連線字串，確保安全性。

## Tech Stack (技術棧)
- Language: Python 3.11
- Orchestration: Python OOP (Pipeline Pattern)
- AI & Parsing: Google Gemini API, EasyOCR, OpenCV
- Web Automation: Selenium, BeautifulSoup4
- Infrastructure: Supabase (PostgreSQL), Google Maps API

## Project Structure (專案結構)

```Plaintext
.
├── etl_pipeline.py             # 主程式 (Controller)
├── crawler_songshan_class.py   # 爬蟲模組 (含 OCR/AI 邏輯)
├── crawler_tfam_class.py       # 爬蟲模組 (含 Selenium 邏輯)
├── ... (其他場館爬蟲)
├── requirements.txt            # 相依套件
└── .env                        # 環境變數 (需自行建立)
```

## Setup & Usage
1. Clone Repository

```bash
    git clone [https://github.com/RayWang98/exhibition_pipline_daily.git]
```

2. Install Dependencies

```bash
    pip install -r requirements.txt
```
3. Configure Environment Create a .env file:

```bash
    DATABASE_URL=postgresql://user:pass@host:5432/db
    GEMINI_API_KEY=your_gemini_key
    GOOGLE_MAPS_API_KEY=your_maps_key
```

4. Run Pipeline

```bash
    python etl_pipeline.py
```

## Author
RayWang
 - Try to learn and transition to Data Engineer.
 - Data Engineer focusing on building robust ETL pipelines for unstructured data.
 - Transforming complex web data into actionable insights.