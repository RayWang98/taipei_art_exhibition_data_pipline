from sqlalchemy import create_engine
from dotenv import load_dotenv
from typing import Dict, List, Tuple # 資料格式定義
import os
import pandas as pd
import streamlit as st



class connectsql_get_data:
    
    def __init__(self):
        self.databasename = os.getenv('databasename')
        self.databasename_tag = os.getenv('databasename_tag')
        self.DATABASE_URL = os.getenv('DATABASE_URL')
        self.SQLQUERY = f'select * from {self.databasename}'
        self.SQLQUERY_TAG = f'select * from {self.databasename_tag}'

    # 使用 Streamlit 的快取機制，避免每次互動都重新查詢資料庫
    # ttl=600 表示每 600 秒 (10 分鐘) 才重新查詢一次資料庫
    @st.cache_data(ttl = 3600, show_spinner = '⏳ 正在建立連線並讀取資料，請稍候...')
    def connectsql_get_data(_self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        load_dotenv()
        if not _self.DATABASE_URL:
            return pd.DataFrame(), pd.DataFrame()
        
        try:
            # 1. 建立 SQLAlchemy 引擎
            engine = create_engine(_self.DATABASE_URL)

            df = pd.read_sql_query(_self.SQLQUERY, engine) # 使用 Pandas 讀取數據
            # 確保坐標是 float 類型並移除 NaN 
            if 'lat' in df.columns and 'lon' in df.columns:
                df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
                df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
                df = df.dropna(subset=['lat', 'lon']) 
            
            df_tag = pd.read_sql_query(_self.SQLQUERY_TAG, engine)
            df_tag['update_flg'] = pd.to_datetime(df_tag['update_flg'])

            return df, df_tag

        except Exception as e:
            print(f'{e}')
            return pd.DataFrame(), pd.DataFrame()
    
    def _translate_date(self, df : pd.DataFrame) -> pd.DataFrame:
        df['update_flg'] = pd.to_datetime(df['update_flg']) + pd.Timedelta(hours = 8)
        df['start_date'] = pd.to_datetime(df['start_date']).dt.strftime('%Y-%m-%d')
        df['end_date'] = pd.to_datetime(df['end_date']).dt.strftime('%Y-%m-%d')
        df.columns = ['展館名稱', '展覽地點', '展覽名稱', '開始日期', '結束日期', '參觀時間', '票價', '緯度', '經度', '網頁連結', '圖片連結', '展覽介紹', '更新時間']
        return df   

def get_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    temp = connectsql_get_data()
    df_exhibitions, df_tags = temp.connectsql_get_data()
    df_exhibitions = temp._translate_date(df_exhibitions)
    df_future_venue = pd.read_csv('taipei_museums_info.csv', sep = ',', encoding = 'utf-8-sig')

    return df_exhibitions, df_tags, df_future_venue