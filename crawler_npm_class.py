# ====================================================================
# I. 核心 AI 服務 (Gemini/RAG) & OCR
# ====================================================================
import cv2                        # 處理圖片 (OCR)
import json                       # 處理 JSON 格式
from google import genai          # Gemini API
from google.genai import types    # Gemini 結構化輸出 Schema
import easyocr                    # OCR 圖片文字識別
from google.genai.errors import APIError # 處理 Gemini API 錯誤
import googlemaps
from googlemaps.exceptions import ApiError, Timeout 

# ====================================================================
# II. 網路請求 (Web/HTTP) 與解析
# ====================================================================
from bs4 import BeautifulSoup as bs           # HTML 解析
import requests as req                        # HTTP 請求
from urllib.parse import urljoin, urlparse    # URL 組合
from urllib3.exceptions import InsecureRequestWarning
import urllib3

# ====================================================================
# III. 檔案/系統與環境變數管理
# ====================================================================
import os                         # 讀取環境變數
import re                         # 正規表達式
from pathlib import Path as pp    # 檔案路徑操作
import time                       # 執行延遲
from dotenv import load_dotenv    # 環境變數

# ====================================================================
# IV. 資料處理、時間處理與隨機性變數
# ====================================================================
import random as rd               # 隨機延遲時間
from datetime import datetime     # 處理日期與時間
import pandas as pd               # 資料處理轉換
import numpy as np                # 資料處理轉換
from typing import Optional, List, Dict, Any # 資料格式定義
from dataclasses import dataclass, field

@dataclass
class exhibition_data:
    # 文字爬蟲資訊
    hallname : str = '國立故宮博物院'
    space : str = '' # 展館的場地位置
    start_date : str = '' # 開始日期
    end_date : str = '' # 結束日期
    title : str = '' # 展覽名稱
    overview : str = '' # 展覽重點敘述
    pageurl : str = '' # 展覽內容網址
    # pageimgurl : List[str] = field(default_factory = list) # 爬蟲取得的圖片 URL 列表，各瀏覽頁面內的每一張圖片
    pagetext : str = ''     # 網頁及 OCR 整合後的文本 # 展覽內容敘述
    big_img_url : str = '' # 首頁上面的展示圖，後續用這個!!!!!
    # big_img_bytes : Optional[bytes] = None     # 存放圖片二進制內容 (在 ETL 流程中可能需要暫存) # 這裡放圖片的二進制內容，等等每個都放進去辨識內容，取出有用資訊
    # 圖片及AI整合後資訊
    visit_time_interval : str = '無資訊'
    price : str = '無資訊'
    note : str = '無資訊'
    url : List[str] = field(default_factory = list) # 設定不可變動的預設值，特別是當欄位的預設值是可變容器
    lat : Optional[float] = None # 緯度
    lon : Optional[float] = None # 經度
    addr: str = ''

    # ======================================
    # field(default_factory=list) 的作用是告訴 Python：
    # >>「每次建立新的 MutableContainerExample 實例時，不要共享舊的 list 物件，請呼叫 list() 這個工廠函式，
    # >> 為這個新的實例建立一個全新的、獨立的 list 物件。」
    # ======================================

class EmptyResponseError(Exception):
    '''自定義錯誤：當 API 回傳空的文字內容時拋出。'''
    pass

class ExhibitionETLPipeline:
    urllib3.disable_warnings(InsecureRequestWarning)
    def __init__(self):
        print('Info: 初始化 ETL Pipeline...')
        # 環境變數與設定
        USER_AGENT = os.environ.get('USER_AGENT')
        CONNECTION = os.environ.get('CONNECTION', 'keep-alive')
        ACCEPT = os.environ.get('ACCEPT', '*/*')
        self.hd = {'user-agent': USER_AGENT, 'connection' : CONNECTION, 'accept' : ACCEPT} # 瀏覽器agent設定
        self.GEMINI_KEY = os.getenv('GEMINI_API_KEY') # AI APIKEY
        self.urlpath = r'https://www.npm.gov.tw/Exhibition-Current.aspx?sno=03000060&l=1' # 爬取首頁
        self.pricepath = r'https://www.npm.gov.tw/Articles.aspx?sno=02007004&l=1' # 票價網頁
        self.time_interval_url = r'https://www.npm.gov.tw/Articles.aspx?sno=02007001&l=1' # 開館時間網頁
        self.MAX_RETRIES = 5 # AI辨識時最大處理次數設定
        self.INITIAL_DELAY = 10 # 剛開始的等待秒數
        self.MAPS_KEY = os.getenv('GOOGLE_MAPS_API_KEY')
        
        # 初始化外部服務
        # ======================== Gemini ========================
        if self.GEMINI_KEY:
            self.client = genai.Client(api_key = self.GEMINI_KEY)
            print('Info: ======== Gemini 初始化成功')
        else:
            self.client = None
            print('Error: ======== Gemini 初始化失敗')   
        
        # ======================== GOOGLE MAPS ========================
        if self.MAPS_KEY:
            self.gmaps = googlemaps.Client(key = self.MAPS_KEY)
            print('Info: ======== google map初始化成功')
        else:
            self.gmaps = None
            print('Error: ======== google map初始化失敗')

    # 2. Worker Method =========================================================================
    def _get_price_info(self) -> str: # 用來擷取票價資訊
        # 展覽票價連結
        resp = req.get(self.pricepath, headers = self.hd, timeout = rd.randint(10, 20), verify = False)
        soup_p = bs(resp.text, 'html.parser')
        pricesoup = soup_p.select('div.visit-content-title', limit = 5) 

        price_list = []
        for i in pricesoup:
            temp = i.get_text(strip = True)
            if temp == '免費':
                cont = '未滿18歲、身心障礙及陪同者、65歲以上長者、低收入戶、學生及其他相關協會會員等 ' + temp
                price_list.append(cont)

            elif temp != '第一展覽館':
                price_list.append(temp)
        price_info = ' | '.join(price_list) # 組合擷取出來的資訊

        return price_info
    
    def _get_timeinterval_info(self) -> str: # 用來開閉館時間
        # 參觀時間連結
        restime = req.get(self.time_interval_url, headers = self.hd, timeout = rd.randint(20, 30), verify = False)
        soup_tv = bs(restime.text, 'html.parser')
        visit_time_interval = soup_tv.select_one('div.visit-content p').get_text(strip = True)

        return visit_time_interval
    
    def _get_addr_info(self, sp : bs) -> str:
        addr = sp.select_one('div.flex.items-start.lg\\:items-center.mt-2 p').get_text(strip=True)
        return addr
    
    def _get_overview_content(self, upath : str) -> str:
        res = req.get(upath, headers = self.hd, timeout = rd.randint(20, 30), verify = False)
        soup = bs(res.text, 'html.parser')
        content_bs = soup.select_one('div#section1 h3.h3')
        url_overview = []
        if content_bs:
            tsoup = soup.select_one('div#section1 h3.h3').find_next_siblings('p')
            for i in tsoup:
                if '相關出版品' in i.get_text(strip = True):
                    break
                else:
                    url_overview.append(i.get_text(strip = True))
        return ''.join(url_overview)

    def _extract_base_info(self) -> List[exhibition_data]: # 用來抓取每個展覽的細項
        extracted_data : List[exhibition_data] = []
        res = req.get(self.urlpath, headers = self.hd, timeout = rd.randint(10, 20), verify = False) # 讀取展覽頁的內容
        soup = bs(res.text, 'html.parser')
        hreflist = soup.select('div.navtabs-content-static li.mb-8') # 取出每個展覽的框格；包含名稱、日期、地點等

        for ccnt, i in enumerate(hreflist):
            # 取得網頁回應            
            data = exhibition_data() # 建立data的class，後續調用相關屬性使用
            data.pageurl = urljoin(self.urlpath, i.select_one('a').get('href'))

            title_curt = i.select_one('div.card-content-top h3.font-medium') # 有期限的展覽
            title_usa = i.select_one('div.card-content-top h3.card-title') # 常設展覽
            data.title = title_curt.get_text(strip = True) if title_curt else title_usa.get_text(strip = True) # 展覽名稱 # type: ignore

            date_curt = i.select_one('div.card-content-top div.mt-4') # 有期限的展覽日期位置
            date_usa = date_curt if title_curt else i.select_one('div.card-content-top h3.card-title').find_next_sibling('div') # 常設展的展覽日期位置
            start_date = date_curt.get_text(strip = True).split(r'~')[0] if date_curt else date_usa.get_text(strip = True)
            end_date = date_curt.get_text(strip = True).split(r'~')[1] if date_curt else date_usa.get_text(strip = True)
            data.start_date = '1990-01-01' if not start_date else start_date.replace('常設展', '1900-01-01') # 展覽開始日期
            data.end_date = '2050-12-31' if not end_date else end_date.replace('常設展', '2050-12-31') # 展覽開始日期

            data.overview = self._get_overview_content(urljoin(self.urlpath, i.select_one('a').get('href'))) # 展覽內頁說明
            data.addr = self._get_addr_info(soup) # 館址地址
            data.price = self._get_price_info() # 票價
            
            # 圖片 URL 處理：這裡只存 big_img 的 URL，實際圖片下載會在後續步驟
            data.big_img_url = urljoin(self.urlpath, i.select_one('figure.card-image img').get('data-src')) # 抓取展覽大圖
            data.space = i.select_one('div.card-content-bottom div').get_text(strip = True)
            data.visit_time_interval = self._get_timeinterval_info() # 可參觀時間

            print(f'[{ccnt + 1}/{len(hreflist)}] 提取基礎資訊: {data.title}') 
            extracted_data.append(data)
            time.sleep(rd.randint(1, 15))
            
        return extracted_data
    
    def _transform_googlegeocoding(self, anns : List[exhibition_data]) -> List[exhibition_data]: # 透過googleAPI取回座標
        if not self.gmaps:
            return anns
        print('Info : 執行地理編碼')

        for item in anns:
            full_addr = f'{item.addr}'

            try:
                geocode_result = self.gmaps.geocode(full_addr) # type: ignore
                if geocode_result:
                    location = geocode_result[0]['geometry']['location']
                    item.lat = location['lat']
                    item.lon = location['lng']
                    print(f'Success: {item.title} -> ({item.lat}, {item.lon})')
                else:
                    print(f'Warning: 找不到地址座標 -> {full_addr}')
                    item.lat = 0.0
                    item.lon = 0.0
            except Exception as e:
                print(f'Error: Geo-coding 錯誤 ({item.title}): {e}')
            
            # 雖然 Google 速限很高，但保持禮貌稍微停頓
            time.sleep(0.1)
        return anns

    # 3. Execution Method (這是核心 ETL 流程，取代 main() 邏輯)
    def run_pipeline(self) -> pd.DataFrame:
        cwddate = datetime.strftime(datetime.today(), '%Y%m%d')
        print(f'=== 開始 ETL 流程，資料日期 {cwddate} ===')

        # I. 提取網頁資訊
        anns = self._extract_base_info()

        # II. Geo-Coding 轉換
        anns = self._transform_googlegeocoding(anns)
        
        # III. 載入 (Load) - 未來您的 DTL 函式
        savedata = [item.__dict__ for item in anns] # 將 dataclass 轉為 dict 列表，並創建 DataFrame
        final_df = pd.DataFrame(savedata)
        
        return final_df


# # 4. 檔案運行入口 (單一檔案測試用)；僅保留作為單機版測試用
# if __name__ == '__main__':
#     load_dotenv()
#     pipeline = ExhibitionETLPipeline()
#     final_df = pipeline.run_pipeline()

