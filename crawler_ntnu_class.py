# ====================================================================
# I. 核心 AI 服務 (Gemini/RAG) & OCR
# ====================================================================
import json                       # 處理 JSON 格式
from google import genai          # Gemini API
from google.genai import types    # Gemini 結構化輸出 Schema
from google.genai.errors import APIError # 處理 Gemini API 錯誤
import googlemaps
from googlemaps.exceptions import ApiError, Timeout 

# ====================================================================
# II. 網路請求 (Web/HTTP) 與解析
# ====================================================================
from bs4 import BeautifulSoup as bs           # HTML 解析
import requests as req                        # HTTP 請求
from urllib.parse import urljoin, urlparse    # URL 組合

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
from typing import Optional, List, Tuple # 資料格式定義
from dataclasses import dataclass, field

@dataclass
class exhibition_data:
    # 文字爬蟲資訊
    hallname : str = '國立師大美術館'
    title : str = '' # 展覽名稱
    addr: str = '' # 展館地址
    space : str = '' # 展館的場地位置
    pageurl : str = '' # 展覽內容網址
    # pageimgurl : List[str] = field(default_factory = list) # 爬蟲取得的圖片 URL 列表，各瀏覽頁面內的每一張圖片
    pagetext : str = ''     # 網頁的文本 # 展覽內容敘述，要請Gemini分辨出日期、地點及內文等資訊
    big_img_url : str = '' # 首頁上面的展示圖，後續用這個!!!!!
    # big_img_bytes : Optional[bytes] = None     # 存放圖片二進制內容 (在 ETL 流程中可能需要暫存) # 這裡放圖片的二進制內容，等等每個都放進去辨識內容，取出有用資訊
    # 圖片及AI整合後資訊
    visit_time_interval : str = '無資訊'
    price : str = '無資訊'
    note : str = '無資訊'
    lat : Optional[float] = None # 緯度
    lon : Optional[float] = None # 經度
    start_date : str = '' # 開始日期
    end_date : str = '' # 結束日期
    overview : str = '' # 展覽重點敘述

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
        self.GEMINI_KEY = os.getenv('GEMINI_API_KEY') # AI APIKEY
        self.urlpath = r'https://www.artmuse.ntnu.edu.tw/index.php/current_exhibit/' # 爬取首頁
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


    # 1. Worker Method =========================================================================
    def _get_addrandtimeinterval_info(self) -> Tuple[str, str]:
        # 展覽頁面連結
        res = req.get(self.urlpath, headers = self.hd, timeout = 12)
        soup = bs(res.text, 'html.parser')
        
        addr = soup.select_one('div.textwidget div').find_next_sibling('div').get_text(strip = True).split('：')[-1]
        timeinterval = []
        for i in soup.select('div#block-9 p')[1:]:
            timeinterval.append(i.get_text(strip = True).replace('開放時間：', ''))

        return addr, ' | '.join(timeinterval)
    
    def _extract_base_info(self, timeinterval : str, addr : str) -> List[exhibition_data]: # 用來抓取每個連結的細項
        extracted_data : List[exhibition_data] = []
        res = req.get(self.urlpath, headers = self.hd, timeout = 12)
        soup = bs(res.text, 'html.parser')
        hreflist = soup.select('div.elementor-widget-container figure.wp-caption')

        for ccnt, i in enumerate(hreflist):
            data = exhibition_data() 
            data.title = i.select_one('figcaption.widget-image-caption.wp-caption-text').get_text(strip = True)
            data.big_img_url = i.select_one('a img').get('src') # 抓取展覽大圖
            data.visit_time_interval = timeinterval
            data.pageurl = i.select_one('a').get('href')
            data.addr = addr

            # 取得網頁回應
            resi = req.get(data.pageurl, self.hd, timeout = 12)
            soup_details = bs(resi.text, 'html.parser')
            
            data.pagetext = soup_details.select_one('div#primary').get_text(strip = True, separator = '\n')

            print(f'[{ccnt + 1}/{len(hreflist)}] 提取基礎資訊: {data.title}') 
            extracted_data.append(data)
            time.sleep(rd.randint(1, 15))
            
        return extracted_data
    
    def _extract_with_gemini(self, anns: List[exhibition_data]) -> List[exhibition_data]: # 用來跑google gemeni的
        
        # 提示詞與 Schema 定義 (與您原程式碼 V. 保持一致，只是移動到這裡)
        extraction_schema = types.Schema(
        type = types.Type.OBJECT, # 使用 OBJECT 作為單一活動的容器
            properties = {
                'start_date': types.Schema(type = types.Type.STRING, description = '展覽開始日期。'),
                'end_date' : types.Schema(type = types.Type.STRING, description = '展覽結束日期。'),
                'overview' : types.Schema(type = types.Type.STRING, description = '展覽內容說明。請提供該展覽主軸的完整敘述，長度至少 100 字。。'),
                'addr' : types.Schema(type = types.Type.STRING, description = '展覽地址。'),
                'space' : types.Schema(type = types.Type.STRING, description = '展覽場地。')
            },
        required = ['start_date', 'end_date', 'overview', 'addr', 'space'] # 必要欄位
        )

        # 提示詞工程
        base_prompt = r'''
        您是一位專業的數據分析師。您的任務是從提供的單一文本內容中，識別並嚴格提取活動的**開始日期**、**結束日期**、**展覽內容**、**展覽場地**、**展覽地點**資訊。

        [待分析文本]
        {WEB_TEXT}

        [RAG 參考資訊]
        請參考這個展覽的網頁基礎資訊，作為比對上下文：{名稱: {TITLE}、地址: {ADDR}}

        [輸出規則]
        1. 請將提取的結果封裝為單一個 JSON 物件，並遵循我指定的 JSON Schema 格式。
        2. 請只返回 JSON 格式的內容，不要有任何多餘的解釋或文字。
        3. 您的主要任務是從文本中找到**所有**的開始日期、結束日期、展覽內容、展覽場地。
        4. 展覽內容必須包含對該展覽主題的完整描述，請盡量擷取主要敘事內容，並排除網站導航、聯繫方式等非敘事性文字。
        5. 請同步比對{ADDR}資訊，有些**展覽場地**內容會附加上展館地址，如果和{ADDR}相同，則**排除地址**，僅保留**場地**的資訊。
        6. 日期要拆分成**開始**及**結束**，如果沒有**結束**，則押上**2050-12-31**；如果沒有**開始**，則押上**1990-01-01**。
        7. **所有日期必須採用 YYYY-MM-DD 的 ISO 8601 標準格式 (例如：2025-12-31)。**
        '''

        # 請模型再思考用
        CORRECTION_PROMPT = '''
        \n[!!!] 警告：您上一次的輸出無法被解析為有效的 JSON 格式。
        請您**嚴格**重新檢查您的輸出內容，並確保它是一個**純淨、完整且符合 JSON 規範**的 JSON 字串
        (有沒有可能是少了上下中括弧或是逗點而已，請您注意這點)，請不要包含任何額外的解釋性文字或引言。謝謝！'''
        cantcatch = []

        # 文本分析迴圈 (使用 self.client, self.MAX_RETRIES, self.INITIAL_DELAY)
        for item in anns:
            current_delay = self.INITIAL_DELAY
            curt_name = item.title 

            # 提示詞外，附上列表上的基礎資訊當作判斷依據 內部RAG工程
            full_prompt = (base_prompt.
                           replace('{TITLE}', item.title).
                           replace('{ADDR}', item.addr).
                           replace('{WEB_TEXT}', item.pagetext)
            )

            for attempt in range(self.MAX_RETRIES): 
                try:
                    print(f'Info : 開始嘗試提取「{curt_name}」活動資訊 (第 {attempt + 1}/{self.MAX_RETRIES} 次)')

                    response = self.client.models.generate_content(
                        model = 'gemini-2.5-flash-lite', 
                        contents = full_prompt, 
                        config=types.GenerateContentConfig( # 設定模型如何回應，包括輸出格式、限制和創造性程度等
                            response_mime_type = 'application/json', # 返回json格式資料
                            response_schema = extraction_schema, # 回應的格式按照前面定義的輸出
                            max_output_tokens = 1024, # 限制回傳的token數量，約3-4個英文字母或半個中文字等於1個token
                            temperature = 0.2 # 愈低的值代表模型的回答更具決定性、準確和可預測，適合需要嚴格數據提取和遵循格式的任務。較高的值則適用於寫作、創意或頭腦風暴。
                        )
                    )
                    # 增加一項檢查：確保 response.text 是個字串
                    if response is None or response.text is None:
                        raise EmptyResponseError(f'Error : API 返回了空的文字內容。')
                    
                    # 如果成功，跳出重試循環 ==================================================== 到這步代表有抓到資料
                    extracted_json = json.loads(response.text)  # dtype dict
                    
                    # 將結果從 JSON 字典寫入 ExhibitionData 的欄位中
                    item.space = extracted_json.get('space', '無資訊')
                    item.start_date = extracted_json.get('start_date', '1990-01-01')
                    item.end_date = extracted_json.get('end_date', '2050-12-31')
                    item.overview = extracted_json.get('overview', '無內容說明')

                    print(f'Successed : 「{curt_name}」成功提取：{item.title}，內容為{extracted_json}')
                    time.sleep(rd.randint(5, 15))
                    break
                    
                except (json.JSONDecodeError, EmptyResponseError) as e:
                    # ... 錯誤處理邏輯 (保持不變) ...
                    # 這裡省略錯誤處理細節，但邏輯與您原先程式碼一致
                    if attempt < self.MAX_RETRIES - 1:
                        print(f'Waring : 警告：=== 「{curt_name}」 === 模型未返回有效 JSON (錯誤：{e})。這次取得的內容是這些  {response.text}')
                        print(f'Action : 要求模型自我修正... 等待 5 秒後重試。')
                        full_prompt += CORRECTION_PROMPT # 將修正指令附加到提示詞中，請模型重新思考可能錯誤的地方
                        time.sleep(5)
                        continue # 繼續下一次重試 (帶著修正提示)
                    else:
                        print(f'Fail : JSON 格式錯誤已達最大重試次數，跳過此項目。')
                        cantcatch.append(curt_name) 
                        time.sleep(rd.randint(5, 15))
                        break

                except APIError as e:
                    if attempt < self.MAX_RETRIES - 1 and 'UNAVAILABLE' in str(e):
                        print(f'Error : 伺服器過載 (503 錯誤)。等待 {current_delay} 秒後重試...')
                        time.sleep(current_delay)
                        current_delay *= 1.5
                        continue
                    else:
                        print(f'Fail : API 呼叫失敗，已達最大重試次數，或發生不可恢復錯誤: {e}')
                        cantcatch.append(curt_name)
                        time.sleep(rd.randint(5, 15))
                        break
                except Exception as e:
                    print(f'Error : 「{curt_name}」發生未知錯誤: {e}')
                    cantcatch.append(curt_name)
                    time.sleep(rd.randint(5, 15))
                    break
            print('***************************')
        print(f'這些是沒有抓到的展覽: {cantcatch}')
        return anns
    
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
        addr, timeinterval = self._get_addrandtimeinterval_info()
        anns = self._extract_base_info(timeinterval, addr)

        # II. AI 結構化提取 (Gemini)
        anns = self._extract_with_gemini(anns) 

        # III. Geo-Coding 轉換
        anns = self._transform_googlegeocoding(anns)
        
        # IV. 載入 (Load) - 未來您的 DTL 函式
        savedata = [item.__dict__ for item in anns] # 將 dataclass 轉為 dict 列表，並創建 DataFrame
        final_df = pd.DataFrame(savedata)
        
        return final_df


# # 4. 檔案運行入口 (單一檔案測試用)；僅保留作為單機版測試用
# if __name__ == '__main__':
#     load_dotenv()
#     pipeline = ExhibitionETLPipeline()
#     final_df = pipeline.run_pipeline()

