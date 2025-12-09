from google import genai          # Gemini API
from google.genai import types    # Gemini 結構化輸出 Schema
from google.genai.errors import APIError # 處理 Gemini API 錯誤
from typing import Optional, List, Dict, Any, Tuple # 資料格式定義
from dataclasses import dataclass, field
import os                         # 讀取環境變數
import time                       # 執行延遲
from dotenv import load_dotenv    # 環境變數
from datetime import datetime     # 處理日期與時間
import pandas as pd               # 資料處理轉換
import numpy as np                # 資料處理轉換
import json
import random as rd
from sqlalchemy import create_engine
from pytz import timezone


@dataclass
class ExhibitionKeyword: # 修正名稱以區別單個項目
    title : str = ''
    hallname : str = ''
    keywords : List[str] = field(default_factory = list) 

@dataclass
class KeywordsAnalysisResult:
    analysis_results: List[ExhibitionKeyword] = field(default_factory = list)

@dataclass
class supabase_db:
    title : str
    hallname : str
    tag : List
    update_flg : datetime

class EmptyResponseError(Exception):
    '''自定義錯誤：當 API 回傳空的文字內容時拋出。'''
    pass

class geniai:
    def __init__(self):
        load_dotenv()
        self.GEMINI_KEY = os.getenv('GEMINI_API_KEY') # AI APIKEY
        self.INITIAL_DELAY = 5
        self.BATCHSIZE = 15
        self.databasename_get = os.getenv('databasename_get')
        self.databasename_save = os.getenv('databasename_save')
        self.DATABASE_URL = os.getenv('DATABASE_URL')
        self.SQLQUERY = f'select title, hallname, overview from {self.databasename_get}'
        self.TIMEZONE = timezone('Asia/Taipei')
        self.keywordlist = ', '.join([
                "虛實敘事", "未來科技", "AI共創", "沉浸多維", "科技趨勢",
                "智慧生活", "人機創意", "資訊視覺", "生成藝術", "科技人性",
                "次世代媒體", "數位媒介", "虛擬導覽", "AI應用", "體驗設計",
                "電競產業", "競技文化", "藝術美學", "當代藝術", "裝置藝術",
                "雕塑藝術", "新媒體", "視覺語言", "光影藝術", "色彩心理",
                "抽象藝術", "創作脈動", "創作者觀", "情緒藝術", "設計展覽",
                "文創商品", "設計創意", "創新思維", "藝文交流", "平面設計",
                "立體裝置", "繪本創作", "國際藝文", "前衛藝術", "現代藝術",
                "典藏藝術", "藝術價值", "美感教育", "創作培養", "文化具象",
                "文創市集", "生活選物", "日常美學", "在地文化", "記憶敘事",
                "城市記憶", "群體文化", "社會脈絡", "生活美學", "手作療癒",
                "職人精神", "工藝美學", "材質觸感", "手作體驗", "工藝設計",
                "城市美食", "料理文化", "表演藝術", "音樂演出", "戲劇表演",
                "劇場敘事", "聲音藝術", "即興演出", "現場文化", "國際交流",
                "節慶活動", "親子共創", "遊戲教育", "故事活動", "教育科技",
                "互動學習", "動畫藝術", "漫畫文化", "虛擬偶像", "二次元風格",
                "同人文化", "ACG世界", "潮流玩具", "角色延伸", "跨界次文",
                "永續生活", "綠色日常", "生態保育", "氣候議題", "循環設計",
                "社會關懷", "公益創新", "青年行動", "工藝技術", "文物修護",
                "陶瓷工藝", "玉器文化", "收藏價值", "歷史脈絡", "文化傳承"
            ]

        )
        

        # 初始化外部服務
        # ======================== Gemini ========================
        if self.GEMINI_KEY:
            self.client = genai.Client(api_key = self.GEMINI_KEY)
            print('Gemini 初始化成功')
        else:
            self.client = None
            print('Gemini 初始化失敗')

        # ======================== SUPABASE ========================
        self.engine: Any = None
        if self.DATABASE_URL:
            self.engine = create_engine(self.DATABASE_URL) # 使用 SQLAlchemy 建立連線引擎
            print('Info : DB 連線成功')
        else:
            print('Info : DB 連線失敗')


    def _connectsql_get_data(self) -> pd.DataFrame:
        try:
            df = pd.read_sql_query(self.SQLQUERY, self.engine) # 使用 Pandas 讀取數據
            return df
        
        except Exception as e:
            return pd.DataFrame()

    def _extract_with_gemini(self, input_df : pd.DataFrame) -> List[ExhibitionKeyword]: # 用來跑google gemeni的
        
        exhibition_keyword_schema = types.Schema(
            type = types.Type.OBJECT, # 使用 OBJECT 作為單一活動的容器
            properties = {
                'title': types.Schema(type = types.Type.STRING, 
                                      description = '展覽的確切標題。'),
                'hallname': types.Schema(type = types.Type.STRING, 
                                      description = '展館名稱。'),
                'keywords': types.Schema(type = types.Type.ARRAY, 
                                     items = types.Schema(type = types.Type.STRING, description = '關鍵字。'),
                                     description = '由3-5個字組成的10到50個關鍵字列表。')
            },
        required = ['title', 'hallname', 'keywords'] # 必要欄位
        )

        final_schema = types.Schema(
            type = types.Type.OBJECT,
            properties={
                'analysis_results': types.Schema(type = types.Type.ARRAY, 
                                                 items = exhibition_keyword_schema,
                                                 description = '包含所有展覽關鍵字分析結果的列表。')
            }
        )

        # 提示詞工程
        base_prompt = r'''
            你是一個專業的文化分析師，專門從展覽描述中提取核心主題關鍵字。
            你的唯一任務是：根據提供的展覽內容，**嚴格地從以下的 [主題關鍵字清單] 中選出最相關的關鍵字**。

            **[主題關鍵字清單]**：
            {關鍵字列表}
            ---
            **提取規則：**
            1.  每個展覽的關鍵字**必須且只能**嚴格的從上方的 **[主題關鍵字清單]** 中選取。
            2.  選取的數量不限，只要相關就列入。
            3.  禁止生成清單中不存在的任何詞語。

            **以下是待分析的活動內容：**
            '''
        base_prompt = base_prompt.replace(r'{關鍵字列表}', self.keywordlist)
        # 遍歷 DataFrame 創建輸入內容
        for _, row in input_df.iterrows():
            # 確保 'overview' 欄位不是 List，而是單一字串
            overview_text = ''.join(row['overview']) if isinstance(row['overview'], list) else row['overview']
            base_prompt += f"\n\n== 展覽標題: {row['title']} 及 展館名稱: {row['hallname']} ==\n"
            base_prompt += f"   描述: {overview_text}"

        # 文本分析迴圈 (使用 self.client, self.MAX_RETRIES, self.INITIAL_DELAY)
        try:
            response = self.client.models.generate_content( # 一個包著 List[ExhibitionKeyword] 結構的 JSON 格式，是一個API的嚮應物件
                model = 'gemini-2.5-flash-lite', 
                contents = base_prompt, 
                config = types.GenerateContentConfig( # 設定模型如何回應，包括輸出格式、限制和創造性程度等
                    response_mime_type = 'application/json', # 返回json格式資料
                    response_schema = final_schema, # 回應的格式按照前面定義的輸出
                    max_output_tokens = 16384, # 限制回傳的token數量，約3-4個英文字母或半個中文字等於1個token
                    temperature = 0.4 # 愈低的值代表模型的回答更具決定性、準確和可預測，適合需要嚴格數據提取和遵循格式的任務。較高的值則適用於寫作、創意或頭腦風暴。
                )
            )

            # 增加一項檢查：確保 response.text 是個字串
            if response is None or response.text is None:
                raise EmptyResponseError(f'Error : API 返回了空的文字內容。')
            
            # 如果成功，跳出重試循環 ==================================================== 到這步代表有抓到資料
            extracted_json = json.loads(response.text)  # dtype dict 解析產出的資料
            results_list = extracted_json.get('analysis_results', []) # results_list 現在是一個 Python 字典列表 (List[dict])
            output_data = []
            
            for item_dict in results_list:
                output_data.append(
                    ExhibitionKeyword( # 創建 ExhibitionKeyword 實例
                    title = item_dict.get('title', '無標題'), 
                    hallname = item_dict.get('hallname', '無標題'), 
                    keywords = item_dict.get('keywords', [])
                    )
                )

            print(f'Successed : 批量提取完成。共處理 {len(output_data)} 筆展覽。')
            return output_data

        except APIError as e:
            print(f'API Error: 與 Gemini 通訊時發生錯誤: {e}')
            return []
        
        except Exception as e:
            print(f'General Error: 提取或解析時發生錯誤: {e}')
            return []
        
    def _save_data(self, df : pd.DataFrame) -> bool:

        df['update_flg'] = datetime.now(self.TIMEZONE).strftime(r'%Y-%m-%d %H:%M:%S')

        try:
            df.to_sql( 
                name = str(self.databasename_save), 
                con = self.engine, 
                if_exists = 'replace',
                index = False
            )

            print(f'✅ 數據成功累積載入 Supabase 到表格 {self.databasename_save}，共 {len(df)} 筆。')
            db_save_state = True
            
        except Exception as e:
            print(f'❌ 數據載入失敗，錯誤訊息: {e}')
            db_save_state = False
        return db_save_state
    
    # -----------------------------------------------------------------
    # 新增一個執行主線函式來模擬調用
    # -----------------------------------------------------------------
    def run_ai_analysis(self) ->  bool:
        load_dotenv()
        df_raw = self._connectsql_get_data()
        if df_raw.empty:
            print('無展覽數據可供分析。')
            logs = False
            return logs
        
        print(f"總共 {len(df_raw)} 筆展覽等待 AI 分析...")

        all_keywords : List[ExhibitionKeyword] = []
        num_batches = (len(df_raw) + self.BATCHSIZE - 1) // self.BATCHSIZE
        print(f'總共 {len(df_raw)} 筆展覽，將分 {num_batches} 批次進行 AI 分析...')
        
        

        for i in range(num_batches):
            strt_loc = i * self.BATCHSIZE
            end_index = (i + 1) * self.BATCHSIZE
            df_batch = df_raw.iloc[strt_loc : end_index, :]

            batch_keywords : List[ExhibitionKeyword] = self._extract_with_gemini(df_batch) # 創建一個ExhibitionKeyword的DataClass出來
            all_keywords.extend(batch_keywords)  # 將回傳的ExhibitionKeyword的DataClass放進去，因為類型相同，可以直接extend

        savedata : List[dict] = [item.__dict__ for item in all_keywords] # 將 dataclass 轉為 dict 列表，並創建 DataFrame
        final_df : pd.DataFrame = pd.DataFrame(savedata)
        logs = self._save_data(final_df)
        
        return logs
    

# if __name__ == '__main__':
#     load_dotenv()
#     temp = geniai()
#     logs = temp.run_ai_analysis()
