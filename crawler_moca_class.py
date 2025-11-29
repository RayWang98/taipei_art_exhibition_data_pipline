# ====================================================================
# I. 核心 AI 服務 (Gemini/RAG) & OCR
# ====================================================================
from google import genai          # Gemini API
from google.genai import types    # Gemini 結構化輸出 Schema
from google.genai.errors import APIError # 處理 Gemini API 錯誤

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
from pathlib import Path as pp    # 檔案路徑操作
import time                       # 執行延遲
from dotenv import load_dotenv    # 環境變數

# ====================================================================
# IV. 資料處理、時間處理與隨機性變數
# ====================================================================
import random as rd               # 隨機延遲時間
from datetime import datetime     # 處理日期與時間
import pandas as pd               # 資料處理轉換
from typing import Optional, List, Tuple # 資料格式定義
from dataclasses import dataclass, field


@dataclass
class exhibition_data:
    # 文字爬蟲資訊
    hallname : str = '台北當代藝術館'
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
        self.urlpath = r'https://www.moca.taipei/tw/ExhibitionAndEvent/Exhibitions/Current%20Exhibition' # 爬取首頁
        self.priceandtimepath = r'https://www.moca.taipei/tw/Visit/%E6%99%82%E9%96%93%E8%88%87%E7%A5%A8%E5%83%B9' # 票價網頁
        self.MAX_RETRIES = 5 # AI辨識時最大處理次數設定
        self.INITIAL_DELAY = 10 # 剛開始的等待秒數
        self.MAPS_KEY = os.getenv('GOOGLE_MAPS_API_KEY')
        
        # 初始化外部服務        
        # ======================== GOOGLE MAPS ========================
        if self.MAPS_KEY:
            self.gmaps = googlemaps.Client(key = self.MAPS_KEY)
            print('Info: ======== google map初始化成功')
        else:
            self.gmaps = None
            print('Error: ======== google map初始化失敗')

    # 2. Worker Method =========================================================================
    def _get_price_time_addr_info(self) -> Tuple[str, str, str]: # 用來擷取票價資訊
        # 展覽票價連結
        resp = req.get(self.priceandtimepath, headers = self.hd, timeout = rd.randint(10, 20), verify = False)
        soup = bs(resp.text, 'html.parser')
        price_container = soup.select('ul.secBox li.list.spec')

        price_list = []
        for i in price_container:
            pricetype = i.select_one('p.title').get_text(strip = True)
            pricenote = i.select_one('p.note span.num').get_text(strip = True).replace('NT$ ', '') + '元'
            
            if pricetype == '免票對象':
                pricenote = '免費'
                pricetype = r'6 歲以下、身障者、低收入戶、65 歲以上長者、志工、團體等'
                price = pricetype + ' ' + pricenote
            elif pricetype == '特別方案':
                pricenote = '免費'
                pricetype = r'親子日、社區日、學校團體日等特別方案'
                price = pricetype + ' ' + pricenote
            else:
                price = pricetype + ' ' + pricenote
            price_list.append(price)

        visit_time_interval = soup.select_one('div.secBox p.normal').get_text(strip = True)
        addr = soup.select_one('div.left address').get_text(strip = True).split(' ')[0]

        return (' | '.join(price_list)), visit_time_interval, addr
    
    def _get_overview_time_space_info(self, upath : str) -> Tuple[str, str, str]:
        # 每個展覽內有時間，但有的空白，所以有的舊擷取，空白就用展館的營業時間
        res = req.get(upath, headers = self.hd, timeout = rd.randint(20, 30), verify = False)
        soup = bs(res.text, 'html.parser')
        time_interval = soup.select_one('div.dateBox p.time').get_text(strip = True).replace('  - ', ' - ')
        overview = soup.select_one('p.content.desc span.text').get_text(strip = True, separator = '\n')
        space_check = soup.select_one('ul li div.t').get_text(strip = True)
        space = soup.select_one('ul li p.p').get_text(strip = True, separator = ' | ') if space_check == '展覽地點' else ''

        return time_interval, overview, space

    def _extract_base_info(self) -> List[exhibition_data]: # 用來抓取每個展覽的細項
        extracted_data : List[exhibition_data] = []
        res = req.get(self.urlpath, headers = self.hd, timeout = rd.randint(10, 20), verify = False) # 讀取展覽頁的內容
        soup = bs(res.text, 'html.parser')
        container = soup.select('div.list.show') # 取出框格
        linklist = []

        for ccnt, i in enumerate(container):
            # 取得網頁回應            
            data = exhibition_data() # 建立data的class，後續調用相關屬性使用
            data.pageurl = i.select_one('a.link.cg.cgB').get('href')
            linklist.append(data.pageurl)

            data.title = i.select_one('div.titleBox').get_text(strip = True) # 展覽名稱 # type: ignore

            date_container = i.select('div.dateBox div.date')
            data.start_date = date_container[0].select_one('span.year').get_text(strip = True) + '-' + date_container[0].select_one('p.day').contents[0].strip().replace(' / ', '-')
            data.end_date = date_container[1].select_one('span.year').get_text(strip = True) + '-' + date_container[1].select_one('p.day').contents[0].strip().replace(' / ', '-')

            data.overview, visit_time_interval_detail, data.space = self._get_overview_time_space_info(data.pageurl) # 展覽內頁說明
            data.price, visit_time_interval_official, data.addr = self._get_price_time_addr_info() # 票價  參觀時間  館址地址 
            data.visit_time_interval = visit_time_interval_detail if visit_time_interval_detail == '' else visit_time_interval_official
            
            # 圖片 URL 處理：這裡只存 big_img 的 URL，實際圖片下載會在後續步驟
            data.big_img_url = urljoin(self.urlpath, i.select_one('figure.imgFrame').get('src')) # 抓取展覽大圖

            print(f'[{ccnt + 1}/{len(container)}] 提取基礎資訊: {data.title}') 
            extracted_data.append(data)
            time.sleep(rd.randint(1, 15))
            
        return extracted_data
    
    def _transform_googlegeocoding(self, anns : List[exhibition_data]) -> List[exhibition_data]: # 透過googleAPI取回座標
        if not self.gmaps:
            return anns
        print('Info : 執行地理編碼')

        for item in anns:
            full_addr = f'{item.addr} {item.space}'

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


'''
這個檔案是針對「台北當代藝術館 (MOCA)」的專用爬蟲類別，其最大的特色是將票價、時間、地址的爬取獨立成一個步驟，並直接用於填充數據結構：

資料模型：同樣使用 exhibition_data dataclass，並將 hallname 設為 '台北當代藝術館'。

提取策略（多階段）：

I. 爬取固定資訊 (_get_price_time_addr_info)：

訪問 MOCA 官網上專門的「時間與票價」頁面 (self.priceandtimepath)。

直接從該頁面解析出所有票種及價格，並將結果合併為一個格式化字串 (price)。

提取展館的官方開放時間 (visit_time_interval) 和地址 (addr)。

註：由於票價資訊在單獨頁面且結構清晰，無需使用 AI 結構化提取。

II. 爬取展覽列表 (_extract_base_info)：

訪問當期展覽頁面 (self.urlpath)，遍歷展覽列表。

針對每個展覽，提取標題、起訖日期、大圖 URL 等。

內部呼叫 _get_overview_time_space_info(data.pageurl) 獲取單一展覽內頁的詳情（概述、展覽內的時間、地點）。

將上一步驟爬取到的全館票價 (data.price) 和地址 (data.addr) 直接應用於每個展覽的 exhibition_data 實例中。

轉換 (Transform)：

地理編碼 (_transform_googlegeocoding)：使用 Google Maps Geocoding API 將地址轉換為經緯度。

執行流程：run_pipeline() 協調了這些獨立的提取步驟，最後創建 DataFrame。
'''