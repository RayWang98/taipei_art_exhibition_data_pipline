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
    hallname : str = '富邦美術館'
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
# 定義 AI 結構化輸出的數據結構 (與 Gemini Schema 保持一致)
# 雖然我們從 JSON 載入，但這提供了一個清晰的目標結構

class EmptyResponseError(Exception):
    '''自定義錯誤：當 API 回傳空的文字內容時拋出。'''
    pass

class ExhibitionETLPipeline:
    def __init__(self, default_addr : str = r'110台北市信義區松高路79號'):
        print('Info: 初始化 ETL Pipeline...')
        # 環境變數與設定
        USER_AGENT = os.environ.get('USER_AGENT')
        CONNECTION = os.environ.get('CONNECTION', 'keep-alive')
        ACCEPT = os.environ.get('ACCEPT', '*/*')
        self.hd = {'user-agent': USER_AGENT, 'connection' : CONNECTION, 'accept' : ACCEPT} # 瀏覽器agent設定
        self.GEMINI_KEY = os.getenv('GEMINI_API_KEY') # AI APIKEY
        self.urlpath = r'https://www.fubonartmuseum.org/Exhibitions' # 爬取首頁
        self.imgpath = r'https://www.fubonartmuseum.org/FB0202_' # 票價網頁
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

        # ======================== OCR Model ========================
        try:
            self.ocr_reader = easyocr.Reader(['ch_tra', 'en'])
            print('Info: ======== EasyOCR 模型載入成功。')
        except Exception as e:
            print(f'Error: ======== EasyOCR 模型載入失敗: {e}')
            self.ocr_reader = None
        
        # ======================== GOOGLE MAPS ========================
        if self.MAPS_KEY:
            self.gmaps = googlemaps.Client(key = self.MAPS_KEY)
            print('Info: ======== google map初始化成功')
        else:
            self.gmaps = None
            print('Error: ======== google map初始化失敗')


    # 2. Worker Method =========================================================================
    def _download_img(self, pricepath : str) -> bytes | None: # 擷取官網票價資訊
        try:
            resp = req.get(pricepath, headers = self.hd, timeout = 10)
            resp.raise_for_status()  # 如果狀態碼不是 200，就直接丟出錯誤
            # 返回圖片的二進制內容，等等做ocr判讀
            return resp.content  # 成功時回傳圖片存放的完整路徑

        except Exception as e:
            # 如果下載失敗，回傳訊息
            print(f'[{datetime.now()}] {pricepath} -> {e}\n')
        return None  # 失敗就回傳 None
    
    def _eocr_process(self, image_bytes: bytes) -> str:
        '''
        使用 EasyOCR 讀取記憶體中的圖片bytes，並將結果格式化為易於 AI 提取的純文字字串
        '''
        if not image_bytes or self.ocr_reader is None:
            return f'Error 圖片內容為空或 OCR 讀取器未初始化'

        try:
            # 1. 讀取圖片
            nparr = np.frombuffer(image_bytes, np.uint8) # 圖片檔案的原始位元組，透過 np.frombuffer() 轉換成了 NumPy 陣列
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # 使用 OpenCV 從記憶體中解碼圖片；並告訴 OpenCV應該以彩色 (Color) 模式載入圖片。如果圖片是灰階 (Grayscale)，它會被轉換為三通道彩色。
            if image is None:
                return 'Error 錯誤：無法讀取圖片，請檢查記憶體內儲存狀態。'
            h, w, _ = image.shape # 圖片shape:高、寬、顏色；第三者為BGR，維度3，但這裡不需要使用
            # 2. 執行 OCR
            results = self.ocr_reader.readtext(image, detail = 1)

        except Exception as e:
            return f'Error EasyOCR 或 OpenCV 發生錯誤: {e}'

        # 2. **分左右兩邊區塊** 的動態參數設定================================
        # 設置為圖片高度的 2%。如果文字很小，可能需要降低到 0.01。判斷兩個文字塊是否屬於同一行的 Y 軸距離
        row_tolerance = h * 0.016

        # 如果分隔線不在中間，例如左欄佔 60%，則設為 0.6。判斷文字塊屬於左欄還是右欄的分界線
        center_x_factor = 0.5 # 中間分隔線判斷
        center_x = w * center_x_factor
        center_y_factor = 0.5 # 中間分隔線判斷
        center_y = h * center_y_factor

        processed_data = [] # 用來儲存所有已經確定為一行的數據結構

        # 判斷哪些字應該放在一組
        for bbox, text, _ in results: # 座標點(共有四點左上[0]、右上[1]、右下[2]、左下[3]順時針)、文字內容、信心程度
            # 取得邊界框的中心點 Y 和 X 座標，並且把每個bbox和text都拿出來判斷
            y_center = (bbox[0][1] + bbox[2][1]) / 2 # 判斷辨識區塊的Y軸中心點
            x_center = (bbox[0][0] + bbox[1][0]) / 2 # 判斷辨識區塊的X軸中心點

            # 根據 X 座標判斷欄位
            column = 'left' if x_center < center_x else 'right' # 如果該判斷區塊，落在中線左邊，則代表是左側的區塊
            row = 'up' if y_center < center_y else 'down' # 如果該判斷區塊，落在中線上面，則代表是上面的區塊

            # 判斷是否為同一行
            merged = False # 合併開關
            for item in processed_data:
                if abs(y_center - item['y']) < row_tolerance: # 如果Y軸和該輪的區塊Y軸中心小於判斷距離，則判定為同一句話，如果是同一行，將當前的文字塊添加到已存在的item中
                    item['texts'].append({'text': text, 'x': x_center, 'col': column, 'y' : y_center, 'row' : row})
                    merged = True
                    break # 找到應該放到哪一行後，則跳出此輪循環

            if not merged: # 如果迴圈結束都找不到匹配的行，則創建一個新的行
                processed_data.append({
                    'y': y_center,
                    'texts': [{'text': text, 'x': x_center, 'col': column, 'y' : y_center, 'row' : row}]
                }) # 紀錄區塊Y軸中心點、辨識出來的文字內容、區塊X軸中心點、判斷後是left還是right

        # 排序與格式化輸出
        processed_data = sorted(processed_data, key = lambda r : r['y']) # 取出 y值進行排序

        leftup_column_output = ''
        rightup_column_output = ''
        down_row_uptput = ''

        for rows in processed_data:
            # 對同一行列的文字，依 X 座標排序，判定它屬於哪一個行
            leftup_texts_raw = [t for t in rows['texts'] if t['col'] == 'left' and t['row'] == 'up']
            rightup_texts_raw = [t for t in rows['texts'] if t['col'] == 'right' and t['row'] == 'up']
            down_texts_raw = [t for t in rows['texts'] if t['row'] == 'down']
            
            # 先按 X 軸排序，若 X 相同，則用 Y 軸確保垂直順序
            sorted_leftup_texts = sorted(leftup_texts_raw, key=lambda t: (t['x'], t['y']))
            sorted_rightup_texts = sorted(rightup_texts_raw, key=lambda t: (t['x'], t['y']))
            sorted_down_texts = sorted(down_texts_raw, key = lambda t: (t['x'], t['y']))

            # 去除空白，並轉為list
            leftup_texts = [t['text'] for t in sorted_leftup_texts if t['text'].strip() != '']
            rightup_texts = [t['text'] for t in sorted_rightup_texts if t['text'].strip() != '']
            down_texts = [t['text'] for t in sorted_down_texts if t['text'].strip() != '']

            # 將文字各自串接起來，並加上換行符號
            if leftup_texts:
                leftup_column_output += ' '.join(leftup_texts) + '\n'
            if rightup_texts:
                rightup_column_output += ' '.join(rightup_texts) + '\n'
            if down_texts:
                down_row_uptput += ' '.join(down_texts) + '\n'

        # 最終輸出給 AI 的格式
        ocr_text_output = '--- OCR 圖片內容提取結果（左右分欄）---\n'
        ocr_text_output += '\n=== 左上欄內容 (優先閱讀) ===\n' + leftup_column_output.strip()
        ocr_text_output += '\n\n=== 右上欄內容 (次要閱讀) ===\n' + rightup_column_output.strip()
        ocr_text_output += '\n\n\n=== 下面欄位內容 (另外版本) ===\n' + down_row_uptput.strip()
        ocr_text_output += '\n--------------------------------------'
        

        return ocr_text_output    

    def _get_exhibition_urls(self) -> List[str]: # 用來爬要抓的網頁連結
        # 展覽頁面連結
        res = req.get(self.urlpath, headers = self.hd, timeout = 12)
        soup = bs(res.text, 'html.parser')

        infour_now = [urljoin(self.urlpath, i.get('href')) for i in soup.select('div.now_list a.fb-exhibitions-card')] # 等等要爬的所有網頁連結
        infour_upcoming = [urljoin(self.urlpath, i.get('href')) for i in soup.select('div.upcoming_list a.fb-exhibitions-card')]
        infourl = infour_now + infour_upcoming

        # 展覽票價連結
        resp = req.get(self.imgpath, headers = self.hd, timeout = 12)
        soup_p = bs(resp.text, 'html.parser')
        priceurl = soup_p.select_one('#alertModal div.modal-body img').get('src')

        return infourl, priceurl

    def _extract_base_info(self, infourls : List[str]) -> List[exhibition_data]: # 用來抓取每個連結的細項
        extracted_data : List[exhibition_data] = []
        for ccnt, i in enumerate(infourls):
            # 取得網頁回應
            resi = req.get(i, self.hd, timeout = 12)
            soup = bs(resi.text, 'html.parser')
            
            # 建立data的class，後續調用相關屬性使用
            data = exhibition_data() 
            data.pageurl = i
            data.title = soup.select_one('div#exhibition-info-basic h1.font-h1').get_text(strip = True)  # type: ignore
            date_range = soup.select_one('div#exhibition-info-basic h2:nth-of-type(3)').get_text(strip = True).split()
            data.start_date = date_range[0].replace('.', '-')
            data.end_date = date_range[-1].replace('.', '-')
            data.overview = soup.select_one('div.info_detail div.font-body').get_text(strip = True).split('觀展預約')[0]
            data.addr = soup.select_one('div.contact_address p.font-sm').get_text(strip = True, separator = '\n').split('\n')[-1] # type: ignore
            
            # 圖片 URL 處理：這裡只存 big_img 的 URL，實際圖片下載會在後續步驟
            data.big_img_url = soup.select_one('div.fb-photo-frame img').get('src') # 抓取展覽大圖
            data.space = soup.select_one('div.content_location p:nth-of-type(2)').get_text(strip = True)
            data.pagetext = soup.select_one('div.info_detail div.font-body').get_text(strip = True).split('觀展預約')[0] # 同overview
            data.visit_time_interval = soup.select_one('div.contact_schedule p.font-sm').get_text(strip = True, separator = ' ; ')

            print(f'[{ccnt + 1}/{len(infourls)}] 提取基礎資訊: {data.title}') 
            extracted_data.append(data)
            time.sleep(rd.randint(1, 15))
            
        return extracted_data
    
    def _extract_with_gemini(self, anns: List[exhibition_data]) -> List[exhibition_data]: # 用來跑google gemeni的
        
        # 提示詞與 Schema 定義 (與您原程式碼 V. 保持一致，只是移動到這裡)
        extraction_schema = types.Schema(
        type = types.Type.OBJECT, # 使用 OBJECT 作為單一活動的容器
            properties = {
                'title': types.Schema(type = types.Type.STRING, description = '活動的主要名稱或標題。'),
                'price': types.Schema(type = types.Type.STRING, description = '票務或入場資訊，如果免費請寫 **免費入場**。')
            },
        required = ['title', 'price'] # 必要欄位
        )

        # 提示詞工程
        base_prompt = r'''
        您是一位專業的數據分析師。您的任務是從提供的單一文本內容中，識別並嚴格提取活動的票務資訊。
        
        **以下是待分析的活動文本，請特別注意：文本結尾處會包含由 OCR 辨識出的票價資訊，這些資訊以 '--- OCR 圖片內容提取結果 ...' 開頭，請優先從此部分提取票價。**

        [待分析文本]
        {OCR_AND_WEB_TEXT}
        
        [RAG 參考資訊]
        請參考這個展覽的網頁基礎資訊，作為比對上下文：{名稱: {TITLE}、開始時間:{START_DATE}、結束時間:{END_DATE}、展覽說明:{OVERVIEW}、地點:{SPACE}}

        [輸出規則]
        1. 請將提取的結果封裝為單一個 JSON 物件，並遵循我指定的 JSON Schema 格式。
        2. 請只返回 JSON 格式的內容，不要有任何多餘的解釋或文字。
        3. 您的主要任務是從文本中找到**所有**票種及對應價格。請將結果嚴格合併轉為下面格式(用 **|** 連結區隔)：**[票種名稱]:[價格] \| [票種名稱]:[價格]**。
        4. 如果找不到任何票價資訊，請在 price 欄位中填入**無資訊**。
        5. 有一種類別是**免票**，請將該名稱改成該欄位後面的說明，帶票價金額維持0元。例如**未滿6歲之兒童或身心障礙本人**。
        6. 一個展覽只有一個區塊，請判斷後不要將所有區塊加在一起。
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
            rag_info = (
                f'名稱:{item.title}、開始時間:{item.start_date}、結束時間:{item.end_date}、'
                f'展覽說明:{item.overview}、地點:{item.space}'
            )

            # 提示詞外，附上列表上的基礎資訊當作判斷依據 內部RAG工程
            full_prompt = (base_prompt.
                           replace('{OCR_AND_WEB_TEXT}', item.pagetext).
                           replace('{TITLE}', item.title).
                           replace('{START_DATE}', item.start_date).
                           replace('{END_DATE}', item.end_date).
                           replace('{OVERVIEW}', item.overview).
                           replace('{SPACE}', item.space)
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
                    item.price = extracted_json.get('price', '無資訊')

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
        infourl, priceinfo = self._get_exhibition_urls() # 另一個方法來取得網址
        anns = self._extract_base_info(infourl)

        # II. 圖片 OCR 處理
        # 呼叫 self._download_img 和 self.eocr_process
        for idx, item in enumerate(anns):
            # 針對每個展覽，檢查票價資訊
            print(f'=== OCR辨識中 =========== {idx + 1} / {len(anns)} ==== {item.title}')
                
            # 使用下載方法，並傳入 self.imgpath 和 self.hd
            image_bytes = self._download_img(priceinfo)
            
            # 使用 OCR 處理方法
            ocrtext = self._eocr_process(image_bytes) if image_bytes else None
            
            if ocrtext:
                item.pagetext += ('\n接下來是圖片OCR內容文字，' + ocrtext) # 將OCR結果加入anns儲存中了

        # III. AI 結構化提取 (Gemini)
        anns = self._extract_with_gemini(anns) 

        # IV. Geo-Coding 轉換
        anns = self._transform_googlegeocoding(anns)
        
        # V. 載入 (Load) - 未來您的 DTL 函式
        savedata = [item.__dict__ for item in anns] # 將 dataclass 轉為 dict 列表，並創建 DataFrame
        final_df = pd.DataFrame(savedata)
        
        return final_df


# # 4. 檔案運行入口 (單一檔案測試用)；僅保留作為單機版測試用
# if __name__ == '__main__':

#     load_dotenv()
#     pipeline = ExhibitionETLPipeline()
#     final_df = pipeline.run_pipeline()

