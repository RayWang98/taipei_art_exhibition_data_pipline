import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from typing import List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pytz import timezone
# 各展館爬蟲內容
from crawler_songshan_class import ExhibitionETLPipeline as shpipline
from crawler_fubon_class import ExhibitionETLPipeline as fbpipline
from crawler_npm_class import ExhibitionETLPipeline as npmpipline
from crawler_tfam_class import ExhibitionETLPipeline as tfampipline
from crawler_ntnu_class import ExhibitionETLPipeline as ntnupipline
from crawler_moca_class import ExhibitionETLPipeline as mocapipline
from crawler_huashan_class import ExhibitionETLPipeline as huashanpipline

crawler_map = {
    '松山文創園區' : shpipline,
    '富邦美術館' : fbpipline,
    '國立故宮博物院' : npmpipline,
    '臺北市立美術館' : tfampipline,
    '國立師大美術館' : ntnupipline,
    '台北當代藝術館' : mocapipline,
    '華山1914文化創意園區' : huashanpipline

}

@dataclass
class supabase_db:
    sys_id : Optional[str] = field(default_factory =None)
    hallname : str = ''
    space : str = ''
    title : str = ''
    start_date : datetime = field(default_factory = datetime.now)
    end_date : datetime = field(default_factory = datetime.now)
    visit_time_interval : str = ''
    price : str = ''
    lat : Optional[float] = None
    lon : Optional[float] = None
    src_url : str = ''
    img_url : str = ''
    overview : Optional[str] = None
    update_flg : datetime = field(default_factory = datetime.now)


class ExhibitionETLPipeline:
    def __init__(self, VENUE_NAME : List):
    # --- 1. 核心設定 ---
        self.DATABASE_URL = os.getenv('DATABASE_URL') # 從 .env 讀取 Supabase 連線字串
        self.TARGET_TABLE = 'exhibition_data' # 我們在 Supabase 建立的資料表名稱
        self.VENUE_NAME = VENUE_NAME 
        self.TIMEZONE = timezone('Asia/Taipei')
    # --- 2. 啟動相關連線 初始化外部服務 ---
        # Supabase
        self.engine: Any = None
        if self.DATABASE_URL:
            self.engine = create_engine(self.DATABASE_URL) # 使用 SQLAlchemy 建立連線引擎
            print('Info : DB 連線成功')
        else:
            print('Info : DB 連線失敗')

    # --- 3. 數據提取 (Extract) ---
    def _extract_data(self) -> pd.DataFrame:
        '''
        執行爬蟲或 API 呼叫，從單一場館獲取最新的展覽數據。
        Args:
            venue_name: 展館名稱，用於定位爬蟲目標。
        Returns:
            包含原始展覽數據的 Pandas DataFrame。
        '''
        df_all = pd.DataFrame()

        for idx, i in enumerate(self.VENUE_NAME):

            print(f'--- 1. 開始提取 {i} 數據 ---')
            crawlerclass = crawler_map.get(i) 
            if crawlerclass is None:
                continue
            # ----------------------------------------------------
            #  TODO: 獲取數據的爬蟲程式碼放在這裡
            try:
                temp = crawlerclass().run_pipeline()
                if not temp.empty:
                    print(f"提取完成，目前提取{idx + 1} / {len(self.VENUE_NAME)} : {temp['title']} ====== 共 {len(temp)} 筆記錄加入。")
                    df_all = pd.concat([df_all, temp], axis = 0, ignore_index = True)
                    print()
                else:
                    print(f"未找到新數據，跳過 : {temp['title']}")
                    

            except Exception as e:
                print(f'Error: 執行 {i} 爬蟲時發生錯誤: {e}')
            # ----------------------------------------------------
        
        print(f'提取完成。累積共 {len(df_all)}筆記錄')
        return df_all.reset_index(drop = True)

    # 4.數據轉換 (Transform) Data Translation Layer
    def _transform_data(self) -> List[supabase_db]:
        '''
        清洗並標準化原始數據，使其符合 Supabase 資料表的欄位結構。
        '''
        print('--- 2. 開始數據轉換 (DTL) ---')
        
        df_raw = self._extract_data()  # 爬取資料回來
        trans_db : List[supabase_db] = [] # 初始化目標資料模型列表
        
        for idx, row in df_raw.iterrows():  # 將 DataFrame 逐行轉換為 dataclass 實例
            idx : int = idx
            # 建立 supabase_db 實例，並將 DataFrame 的欄位值映射過去
            record = supabase_db( # row 是一個 Series，可以使用 .get() 方法，但通常是 row['key']
                hallname = row['hallname'], # 使用 .get 避免 KeyError
                space = row['space'],
                title = row['title'],
                start_date = pd.to_datetime(row['start_date']),
                end_date = pd.to_datetime(row['end_date']),
                visit_time_interval = str(row['visit_time_interval']),
                price = str(row['price']),
                lat = float(row['lat']),
                lon = float(row['lon']),
                src_url = row['pageurl'], # 假設原始欄位名為 'pageurl'
                img_url = row['big_img_url'],
                overview = row['overview'],
                update_flg = datetime.now(self.TIMEZONE)
            )
            trans_db.append(record)
            print(f"轉換資料{idx + 1}/{len(df_raw)}.{record.hallname} - {record.title}  轉換中==")


        return trans_db

# --- 5. 數據載入 (Load) ---
    def _load_data(self, records : List[supabase_db]) -> None:
        '''
        連接 PostgreSQL 資料庫 (Supabase)，並將數據寫入目標表格。
        '''
        
        df = pd.DataFrame([item.__dict__ for item in records])
        df.drop(columns = ['sys_id'])

        try:
            # 使用 pandas 的 to_sql 寫入數據
            # if_exists='replace'：每次執行都刪除舊表格並新建 (測試用)
            # index=False：不將 DataFrame 的索引寫入資料庫
            df.to_sql(
                name = self.TARGET_TABLE, 
                con = self.engine, 
                if_exists = 'append', # *** 這裡可以改為 'append' 或 'replace' ***
                index = False
            )
            
            print(f"✅ 數據成功載入 Supabase 到表格 {self.TARGET_TABLE}，共 {len(df)} 筆。")
            
        except Exception as e:
            print(f'❌ 數據載入失敗，錯誤訊息: {e}')

# --- 5. 執行主管線 (Pipeline Execution) ---
    def run_pipeline(self):
        '''執行 ETL 數據管線的函式。'''
            
        # 1. 轉換
        df_transformed = self._transform_data()
        
        # 2. 載入
        self._load_data(df_transformed)


if __name__ == '__main__':
    # 載入環境變數（僅用於本地開發調試）
    load_dotenv()
    # 在本地運行時，執行整個 ETL 流程
    main_pip = ExhibitionETLPipeline(['松山文創園區', '富邦美術館', '國立故宮博物院', '臺北市立美術館', '國立師大美術館', '台北當代藝術館'])
    #'松山文創園區', '富邦美術館', '國立故宮博物院', '臺北市立美術館', '國立師大美術館', '台北當代藝術館', '華山1914文化創意園區'
    main_pip.run_pipeline()


'''
專案核心 ETL 流程：這個檔案實現了您資料工程專案的 Core Data Pipeline (ETL) 階段。

資料模型：定義了 supabase_db dataclass 作為您在 Supabase 資料庫中的目標資料模型（Data Translation Layer, DTL）。

提取 (Extract)：_extract_data 方法透過 crawler_map 字典，迭代調用七個展館對應的爬蟲類別 (crawler_*_class.py 中定義的 ExhibitionETLPipeline) 的 run_pipeline() 方法來收集原始資料。

轉換 (Transform)：_transform_data 方法將爬取回來的 Pandas DataFrame 轉換為 supabase_db dataclass 的列表。

載入 (Load) & 資料持久化：_load_data 方法使用 SQLAlchemy 引擎（透過 DATABASE_URL 環境變數連線 Supabase PostgreSQL）將轉換後的數據寫入 exhibition_data 表格中，目前使用 if_exists='replace' 策略。

環境變數管理：使用 os.getenv('DATABASE_URL') 和 dotenv 進行環境變數管理，符合您的規劃。

'''