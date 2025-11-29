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
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options # 導入 Options，因為要自動化執行，所以要採用**無頭模式**
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

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
from typing import Optional, List # 資料格式定義
from dataclasses import dataclass, field

@dataclass
class exhibition_data:
    # 文字爬蟲資訊
    hallname : str = '臺北市立美術館'
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
    def __init__(self):
        print('Info: 初始化 ETL Pipeline...')
        # 環境變數與設定
        USER_AGENT = os.environ.get('USER_AGENT')
        CONNECTION = os.environ.get('CONNECTION', 'keep-alive')
        ACCEPT = os.environ.get('ACCEPT', '*/*')
        self.hd = {'user-agent': USER_AGENT, 'connection' : CONNECTION, 'accept' : ACCEPT} # 瀏覽器agent設定
        self.pricepath = r'https://www.tfam.museum/Common/editor.aspx?id=230&ddlLang=zh-tw' # 爬取首頁
        self.urlpath = r'https://www.tfam.museum/Exhibition/Exhibition.aspx?ddlLang=zh-tw' # 票價網頁
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
        
        # ======================== Webdriver ========================
        try: # 開啟兩個視窗 - 資訊頁 跟 票價頁
            chrome_options = Options() # 設定 Chrome Options
            chrome_options.add_argument('--headless')
            
            # 為了在無頭模式下穩定運行 (尤其在容器或 CI/CD 環境中)
            chrome_options.add_argument('--no-sandbox')         # 禁用沙盒模式，避免在 Linux 環境下出錯
            chrome_options.add_argument('--disable-dev-shm-usage') # 禁用 /dev/shm 內存，尤其在 Docker 中有用
            chrome_options.add_argument('--window-size=1920,1080') # 設定固定視窗大小，避免元素定位問題
            # 避免日誌干擾
            chrome_options.add_argument('--log-level=3') # 設置日誌級別，僅顯示錯誤和致命信息
            self.driver = webdriver.Chrome(
                            service = Service(ChromeDriverManager().install()),
                            options = chrome_options # 傳入設定好的 options
                        )

            # self.driver = webdriver.Chrome(service = Service(ChromeDriverManager().install())) # 載入 Chrome 瀏覽器的驅動服務，並自動下載/安裝正確版本的 WebDriver
            # 展覽清單頁面
            self.driver.get(self.urlpath)
            # self.driver.maximize_window() # 將瀏覽器視窗最大化，確保所有元素都可見（有時是避免元素被遮擋的必要步驟）
            self.main_handle = self.driver.current_window_handle # 儲存主要視窗（展覽清單頁）的控制代碼 (Handle)。 > 這是為了後續在開啟新分頁後，能夠切回這個主視窗。
            # 票價、參觀時間、地址頁面
            self.driver.execute_script("window.open('about:blank','_blank');") # 執行 JavaScript，在瀏覽器中開啟一個空白的新分頁 (_blank)。
            self.driver.switch_to.window(self.driver.window_handles[-1]) # 將 WebDriver 的控制焦點切換到所有視窗控制代碼中的最後一個（即剛開啟的新分頁）。
            self.driver.get(self.pricepath)
            WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, 'body'))) # 設定明確等待 (Explicit Wait)：等待最多 10 秒，直到頁面的 <body> 標籤出現為止。這是為了確保頁面內容已經載入完成，避免抓取到空白頁面。
            print(f'Info: ======== Chrome 初始化開啟成功')
        except Exception as e:
            self.driver = None
            print(f'Error: ======== Chrome 初始化開啟失敗 : {e}')

    # 1. Worker Method =========================================================================
    def _get_visit_time_interval(self, soup : bs) -> str:
        visit_time_interval = soup.select('div.w-2 ul.spacingB-20.mobile li')
        visit_dict = dict()
        for i in visit_time_interval:
            temp = i.get_text(strip = True).split()
            dayofweek = temp[0].replace(':', '')
            time_interval =  ''.join(temp[1:])
            
            if time_interval not in visit_dict:
                visit_dict[time_interval] = [dayofweek]
            else:
                visit_dict[time_interval].append(dayofweek)

        visit_list = []
        for k, v in visit_dict.items():
            if k == '':
                visit_list.append(f'{v[0]}')
            else:
                visit_list.append(f"{','.join(v)} - {k}")

        return (' | '.join(visit_list))
    
    def _get_priceinfo(self, soup : bs) -> str:
        pricetag = []
        for i in soup.select('table.spacingB-20 tbody')[-1].select('tr')[:5]:
            temp = i.get_text(strip = True, separator = '|').replace('\u3000', '')
            if '免費入場' in temp:
                pricetag.append('6 歲以下、身障者、低收入戶、65 歲以上長者、志工、長者、學生、團體:0元')
            else:
                pricetag.append(':'.join(temp.split('|')[:2]) + '元')
        return (' | '.join(pricetag))
    
    def _scroll_to_load_all(self, driver, max_scrolls = 10) -> None:
        # 獲取初始高度
        last_height = driver.execute_script('return document.body.scrollHeight') # 獲取當前網頁內容的總高度 (初始高度)。作為比較的基準，以便檢查是否有新內容載入。
        scroll_count = 0
        
        # 使用 while 迴圈實現捲動與檢查
        while scroll_count < max_scrolls:
            # 捲動到頁面底部
            # 執行 JavaScript，將瀏覽器視窗捲動到頁面的 最底部。模擬使用者向下捲動的動作，觸發網頁的延遲載入機制。 
            # 0代表最左邊X軸的起點，因為轉注於向下捲動，X軸隨便設定沒差別
            driver.execute_script('window.scrollTo(0, document.body.scrollHeight)')      
            time.sleep(1.0)      
            # 獲取新的頁面高度
            new_height = driver.execute_script('return document.body.scrollHeight')  
            # 檢查高度是否停止增加
            if new_height == last_height: # 判斷新高度是否等於舊高度。如果高度沒有變化，表示所有內容都已載入完畢，可以跳出迴圈。
                break
            # 更新舊高度為新高度，準備進行下一次捲動。儲存新的高度，以便下一輪比較。
            last_height = new_height
            scroll_count += 1
        
        if scroll_count >= max_scrolls:
            print(f"警告：已達到最大捲動次數限制 ({max_scrolls} 次)。")

    def _get_exhibition_detail_info(self, driver, driverelement : List, idx : int) -> str:
        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", driverelement[idx]) # 捲動要點擊的項目到視窗中
        before_handles = set(driver.window_handles) # 紀錄視窗
        driverelement[idx].click()

        try:
            # 最多等 10 秒，直到新的分頁出現為止
            (WebDriverWait(driver, 10). # 10是最大等待時間（秒）。在 10 秒內，只要條件為 True，就會立即停止等待並繼續執行；若超時則拋出 TimeoutException。
                until( # 要求 WebDriverWait 檢查傳入的函式（條件）是否返回 True。
                    lambda d: len(set(d.window_handles) - before_handles) > 0 
                    # d 代表傳入的 WebDriver 實例 (即 driver)。 
                    # d.window_handles代表 獲取當前瀏覽器所有開啟視窗或分頁的 Handle 列表
                    # 將當前所有的 Handle 列表轉換成集合，然後減去點擊前記錄的舊 Handle 集合 (before_handles)。取得剩下最新的網頁。
                    # 判斷新 Handle 集合的長度是否大於 0。如果大於 0 (即找到了至少一個新的 Handle)，則條件成立 (返回 True)，等待結束。
                    )
            )
            # 找出唯一的不同 Handle 並切換
            new_tab_handle = list(set(driver.window_handles) - before_handles)[0]
            driver.switch_to.window(new_tab_handle)

        except Exception:
            print(f'無法開啟詳情頁或超時，切回主視窗。')
            driver.switch_to.window(self.main_handle)
            return None, None

        try:
            # 等待詳情頁的關鍵元素出現，替換 time.sleep(10)
            WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, 'div.info-item div.info-content'))
                    # Selenium 內建的一個條件函式 (EC 是 expected_conditions 的縮寫)。
                    # 它要求 WebDriver 等待直到頁面 DOM 中出現指定定位器所指向的元素。
            )
        except:
            # 即使超時，也繼續嘗試提取
            print(f'Info:詳情頁內容載入超時，抓取可能不完整。')

        curt_url = driver.current_url
        temp = bs(driver.page_source, 'html.parser')
        temp_overview = temp.select('div.info-item div.info-content')
        overview = ''.join([i.get_text(strip = True) for i in temp_overview]).split('策展人')[0]
        driver.close()
        driver.switch_to.window(self.main_handle)
        return overview, curt_url

    def _get_addr_info(self, soup : bs) -> str:
        addr = soup.select_one('ul.spacingB-20 li a')
        if addr is None:
            print(addr)
            print(f'Warning : 地址資訊萃取失敗!')
            return ''
        else:
            addr = addr.get_text(strip = True)
        return addr

    def _extract_base_info(self) -> List[exhibition_data]: # 用來抓取每個展覽的細項
        extracted_data : List[exhibition_data] = []
        
        # 地址、票價、開放時間頁面
        price_soup = bs(self.driver.page_source, 'html.parser') # 使用 BeautifulSoup 解析當前頁面（參觀資訊頁）的 HTML 原始碼。接下來的程式碼將使用 info_soup 進行靜態的資料提取。
        addr_info = self._get_addr_info(price_soup)
        visit_time_info = self._get_visit_time_interval(price_soup)
        price_info = self._get_priceinfo(price_soup)
        
        self.driver.close() # 地址、票價、開放時間擷取完成，關閉分頁
        self.driver.switch_to.window(self.main_handle) # 切回展覽頁面
        self._scroll_to_load_all(self.driver) # 滾動頁面，確保所有資訊都引入

        content = self.driver.find_elements(By.CSS_SELECTOR, 'a.ExPage') # 找尋頁面中，各展覽說明細節的頁面連結
        list_soup = bs(self.driver.page_source, 'html.parser') # 轉換為BeautifulSoup物件
        hreflist = list_soup.select('div.row.Exhibition_list') # 取出每個展覽窗格

        for ccnt, i in enumerate(hreflist):          
            data = exhibition_data() # 建立data的class，後續調用相關屬性使用

            data.title = i.select_one('div.w-9 a.ExPage').get_text() # 展覽名稱

            date_interval = i.select_one('div.w-9 p.date-middle').get_text()
            data.start_date = date_interval.split('-')[0] # 展覽開始日期
            data.end_date = date_interval.split('-')[-1] # 展覽開始日期

            data.space = i.select_one('div.w-9 p.info-middle').get_text(strip = True) # 展覽地點
            data.visit_time_interval = visit_time_info # 參觀時間
            data.addr = addr_info # 館址地址
            data.price = price_info # 票價
            data.big_img_url = urljoin(self.urlpath, i.select_one('div.w-8 img').get('src')) # 抓取展覽大圖
            data.overview,  data.pageurl = self._get_exhibition_detail_info(self.driver, content, ccnt) # 展覽內頁說明
            if data.overview is None:
                print(f'Warning : 跳過展覽， {data.title}頁面內，展覽內容說明及網址提取失敗!')
                continue


            print(f'[{ccnt + 1}/{len(hreflist)}] 提取基礎資訊: {data.title}') 
            extracted_data.append(data)
            time.sleep(rd.randint(1, 5))
            
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

動態爬蟲核心 (E)：

Selenium/WebDriver：程式碼使用 selenium 框架，並配置了 Chrome 的無頭模式 (--headless)，確保在沒有圖形介面的環境下也能自動化執行。

多視窗操作：初始化時，它同時開啟兩個分頁：

展覽清單頁 (self.urlpath) - 作為 self.main_handle。

參觀資訊頁（票價、時間、地址）(self.pricepath)。

資訊預先提取：在進入展覽迴圈之前，程式會先從「參觀資訊頁」中提取全館統一的參觀時間 (_get_visit_time_interval)、票價 (_get_priceinfo) 和地址 (_get_addr_info)，然後將這些資訊賦予每一個展覽項目。

捲動載入：_scroll_to_load_all 函數用於模擬用戶捲動到底部，以確保所有採用延遲載入 (Lazy Loading) 的展覽項目都被載入到 DOM 中。

詳情頁處理 (_get_exhibition_detail_info)：

程式碼會點擊展覽連結，觸發新分頁開啟。

使用 WebDriverWait 等待新分頁完全載入，以避免抓取失敗。

在新分頁中提取詳細介紹 (overview) 和最終網址 (pageurl)。

提取完畢後，立即關閉新分頁並切回主頁面 (self.main_handle)，保持主流程的穩定。

資料模型調整：

exhibition_data 的 hallname 設定為 '臺北市立美術館'。

由於票價和時間是統一爬取，資料模型中也新增了 addr 欄位來存放場館地址。

此爬蟲不需要 OCR 或 Gemini 結構化提取，因此相關的匯入和步驟 (例如 pageimgurl, pagetext, _extract_with_gemini) 被移除或略過。

轉換 (T)：

地理編碼 (_transform_googlegeocoding)：使用 Google Maps API 結合場館地址 (item.addr) 和展覽空間 (item.space) 進行地理編碼。'''
