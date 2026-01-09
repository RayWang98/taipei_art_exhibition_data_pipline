# ===================================================
# streamlit  %% app.py
# ===================================================
import os
import pandas as pd
import streamlit as st # 導入 Streamlit 函式庫，用於建構 Web 應用程式介面
from dotenv import load_dotenv
from rapidfuzz import process # 導入 rapidfuzz 函式庫，用於高效的模糊字串匹配
from streamlit.components.v1 import html
import datetime as dt
import json
from typing import Dict, List, Tuple # 資料格式定義
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from matplotlib import font_manager
# 其他功能 =================================================
from io_database import get_data
from recom_sys import recommendation_engine as rec_sys
# =========================================================


class streamlit_run_app:  
    def __init__(self):
        self.config_ttile = '展覽雷達：雙北展覽空間與文化趨勢地圖_Demo'
        self.GOOGLEMAP = os.getenv('GOOGLE_MAPS_API_KEY')
        self.GOOGLEMAPID = os.getenv('GOOGLEMAPID')

        with open('sideprojbrief.txt', 'r', encoding = 'utf-8') as f:
            self.sideprojectbrief = f.read()

        self.topic = r'展覽雷達：雙北展覽空間與文化趨勢地圖'

        self.venue_urls = dict()
        with open('urls.txt', 'r', encoding = 'utf-8') as f:
            for line in f:
                temp : str = line.strip().replace("'", '')
                parts : List = temp.split(',', 1)
                ven_name = parts[0].strip()
                ven_url = parts[1].strip()
                self.venue_urls[ven_name] = ven_url
            
        self.venue_image_urls = dict()
        with open('urls_image.txt', 'r', encoding = 'utf-8') as f:
            for line in f:
                temp : str = line.strip().replace("'", '')
                parts : List = temp.split(',', 1)
                ven_name = parts[0].strip()
                ven_imgurl = parts[1].strip()
                self.venue_image_urls[ven_name] = ven_imgurl

        self.venue_image_urls_src = dict()
        with open('urls_src.txt', 'r', encoding = 'utf-8') as f:
            for line in f:
                temp : str = line.strip().replace("'", '')
                parts : List = temp.split(',', 1)
                ven_name = parts[0].strip()
                ven_imgsrc = parts[1].strip()
                self.venue_image_urls_src[ven_name] = ven_imgsrc

        self.venue_introduction = dict()
        with open('venue_intro.txt', 'r', encoding = 'utf-8') as f:
            for line in f:
                temp : str = line.strip().replace("'", '')
                parts : List = temp.split(',', 1)
                ven_name = parts[0].strip()
                ven_intro = parts[1].strip()
                self.venue_introduction[ven_name] = ven_intro

        self.venue_hashtags : Dict = dict()
        with open('venue_hashtag.txt', 'r', encoding = 'utf-8') as f:
            for line in f:
                temp : str = line.strip().replace("'", '')
                parts : List = temp.split(',', 1)
                ven_name = parts[0].strip()
                ven_hash = parts[1].strip()
                self.venue_hashtags[ven_name] = ven_hash

        # 換頁及所選資訊紀錄初始化
        if 'page_mode' not in st.session_state:
            st.session_state['page_mode'] = 'home' # 預設為首頁
        if 'selected' not in st.session_state:
            st.session_state['selected'] = 'None'
        if 'tag_counts' not in st.session_state:
            st.session_state['tag_counts'] = 0

        self.df_exhibitions, self.df_tags, self.df_future_venue = get_data() # 讀取資料
        self._rec_system = rec_sys() # 取得推薦系統核心

    # def _handle_tag_click(self, ve_name: str, exhibition_id: str):
    #     # 呼叫引擎的方法來記錄點擊
        
    #     # 點擊後，觸發 Streamlit 重新執行，以更新推薦面板和搜尋結果
    #     st.rerun()

    # def _display_tags(self, tags: List[str], exhibition_id: str):
    #     col1, col2 = st.columns([1, 4])
    #     for i, tag in enumerate(tags):
    #         # 使用 unique key
    #         if col1.button(f'#{tag}', key = f'tag_btn_{exhibition_id}_{i}'):
    #             # 點擊時呼叫處理函式
    #             self._handle_tag_click(tag, exhibition_id)
     
    def _display_google_map(self, df: pd.DataFrame, venue_name : str, exhibition_name : str, map_height: int = 700) -> None:
        df_v = df[(df['展館名稱'] == venue_name) & (df['展覽名稱'] == exhibition_name)]
        if df_v.empty:
            st.warning(f'數據庫中找不到屬於 **{df_v}** 的展覽點位。無法顯示地圖。')
            return 
                
        # 準備數據：選取 lat, lon, title 欄位，並轉換為 JSON 格式
        point = df_v[['緯度', '經度', '展覽名稱', '圖片連結', '展覽地點']].to_dict('records')
        point_json = json.dumps(point) # 將 Python 列表轉換為 JavaScript 陣列字串

        # 計算地圖中心點 (所有點的平均值)
        center_lat = df_v['緯度'].mean()
        center_lon = df_v['經度'].mean()

        # Google Maps 的 HTML 和 JavaScript 程式碼
        try:
            # 1. 讀取模板檔案
            with open('google_map_html.html', 'r', encoding='utf-8') as f:
                map_template = f.read()

            # 2. 替換模板變數
            map_html = map_template.replace(
                '{point_json}', point_json
            ).replace(
                '{center_lat}', str(center_lat)
            ).replace(
                '{center_lon}', str(center_lon)
            ).replace(
                '{GOOGLEMAP}', str(self.GOOGLEMAP)
            ).replace(
                '{GOOGLEMAPID}', str(self.GOOGLEMAPID)
            )

            # 3. 使用 Streamlit HTML 元件嵌入地圖  
            html(map_html, height = map_height)

        except FileNotFoundError:
            st.error("錯誤：找不到地圖模板檔案 'map_template.html'。請確認檔案路徑。")
        except Exception as e:
            st.error(f'渲染地圖時發生未知錯誤: {e}')  

    # 🎯 新增函式：使用 st.columns 顯示場館網格列表
    def _display_venue_grid(self, info : pd.DataFrame | dict):
        # 定義每行顯示 4 個欄位 (在寬螢幕下)
        columns = st.columns(4) 

        # 建立容器
        all_venuesorexhibition = [] # 展館名稱 或 展覽名稱
        image_url_dict = dict() # 圖片連結
        img_src_dict = dict()
        hashtags_dict = dict() # 標籤
        clicktext = ''
        page_mode = ''

        # 所有要呈現的列表
        if type(info) == pd.DataFrame:
            src_dict = info[['展覽名稱', '圖片連結', '展覽介紹']].to_dict('records')
            for ids in src_dict:
                all_venuesorexhibition.append(ids.get('展覽名稱'))
                image_url_dict[ids.get('展覽名稱')] = ids.get('圖片連結')
                hashtags_dict[ids.get('展覽名稱')] = ids.get('展覽介紹')[:100] + '...'
                img_src_dict[ids.get('展覽名稱')] = r'# 圖片來源-官網圖片'
                clicktext = r':ghost: 查看展覽說明'
                page_mode = 'exhibition_view'
                
        else:
            all_venuesorexhibition = list(info.keys()) # 首頁用的 home
            image_url_dict = self.venue_image_urls
            hashtags_dict = self.venue_hashtags
            clicktext = r'📍 查看展館中的展覽'
            page_mode = 'map_view'
            src_dict = self.venue_image_urls_src
            img_src_dict = self.venue_image_urls_src
        
        
        
        for i, v_e_name in enumerate(all_venuesorexhibition):
            with columns[i % 4]:
                image_url = image_url_dict.get(v_e_name)
                hashtags = hashtags_dict.get(v_e_name, '')
                src = img_src_dict.get(v_e_name, '')
                
                # 使用 Streamlit 內建的元件來顯示內容
                styled_caption = f"""
                <div style="
                    font-size: 18px; 
                    color: #f4a460; 
                    font-weight: bold; 
                    text-align: left; /* 讓標題置中 */
                    margin-top: 8px; 
                ">
                    {v_e_name}
                </div>
                """
                # 1. 顯示場館圖片
                st.image(
                    image = image_url, 
                    # caption = f'**{v_e_name}**',
                    use_container_width = True, # 讓圖片填滿欄位寬度
                    output_format = 'auto'
                )

                # 2. 顯示 展館名稱
                st.markdown(styled_caption, unsafe_allow_html = True)

                # 3. 顯示 Hashtag 及 圖片來源
                st.markdown(
                    f'<div style="font-size: 12px; color: #888888; margin-top: -1px;">{hashtags}</div>', 
                    unsafe_allow_html = True
                )
                
                st.markdown(
                    f'<div style="font-size: 10px; color: #888888; margin-top: -1px;">{src}</div>', 
                    unsafe_allow_html = True
                )
                
                # 4. 點擊按鈕，實現互動
                # 使用唯一的 key 來區分每個按鈕
                button_key = f'select_{v_e_name}'
                               
                # 如果點擊按鈕，則將場館名稱儲存到 Session State
                if st.button(label = f'**{clicktext}**', key = button_key, use_container_width = True):
                    st.session_state['selected'] = v_e_name
                    st.session_state['page_mode'] = page_mode # 設置頁面模式為地圖視圖
                    st.rerun() 
                    # Button State Lag 或 One-Click Delay ===============================================================================
                    # 第一次點擊，Python 腳本從頭到尾執行了一次。變更session_state 為 **v_e_name**
                    # 第二次點擊，Streamlit 偵測到 Session State 變化，觸發第二次重新執行。
                    # 按鈕邏輯執行完畢並成功更新了 Session State 時，手動強制 Streamlit 立即重新執行(st.rerun())，而不等待 Streamlit 自動處理狀態變化。
                    # ===================================================================================================================

        # 確保 selected 狀態存在
        if 'selected' not in st.session_state:
            st.session_state['selected'] = 'None'
    

    
    # 展館、展覽搜尋功能 =====================================================================
    def _search_fuzzy_wildcard(self, usr_input : str, searchlist : list) -> List[str]:
        choices = [i.lower() for i in searchlist] # 要比對的清單
        
        best_match = process.extract(usr_input.lower(), choices, limit = 3) # 模糊比對，選前三名出來；choices是用戶可選的場館列表
        # 回傳 Tuple：("最佳匹配字串", 分數, 在清單中的 index)

        score_threshold = 45 # 設定分數門檻
        filtered_match_name = [i[0] for i in best_match if i[1] >= score_threshold] # 挑出符合門檻的，其他丟掉

        if filtered_match_name:
            return filtered_match_name
        else:
            return []


    # 數據統計品質功能 =======================================================================

    # =======================================================================================
    # 文字雲功能 - 暫時停止 20251206
    # 影響速度，放到Liu的Tableau平台上面
    # =======================================================================================    
    # def _generate_wordcloud_plot(self, keyword_series : pd.DataFrame) -> None:
    #     # 1. 轉換為頻率字典 {詞彙: 頻率}
    #     word_freq_dict = pd.Series(
    #         keyword_series['出現次數'].values, 
    #         index = keyword_series['Tag']
    #     ).to_dict()

    #     # 2. 定義中文停用詞
    #     custom_stopwords = set([
    #         '的', '是', '在', '與', '和', '展', '覽', '藝術', '作品', '設計', '活動',
    #         '透過', '觀眾', '系列', '個', '由', '於', '為', '將', '年', '代', '日', '{', '}', ','
    #     ])
            
    #     try:
    #         # 4. 初始化 WordCloud 物件
    #         font_path = 'fonts/NotoSansTC-Regular.ttf' # src/fonts/NotoSansTC-Regular.ttf
    #         wordcloud = WordCloud(
    #             font_path = font_path,
    #             width = 2000, 
    #             height = 600,
    #             background_color = None,
    #             mode = 'RGBA', # 設置為 RGBA 模式以支援透明度
    #             max_words = 50,
    #             # stopwords = custom_stopwords,
    #             collocations = False,
    #             prefer_horizontal = 0.9,
    #             colormap = 'Paired'
    #         ).generate_from_frequencies(word_freq_dict) # 注意：這裡使用 generate_from_frequencies

    #         # 5. 使用 Matplotlib 繪圖
    #         fig, ax = plt.subplots(figsize = (20, 15), facecolor = 'none') # facecolor='none' 透明

    #         # 設定 Matplotlib 圖表和軸的背景為透明 (透明度 alpha = 0)
    #         fig.patch.set_alpha(0)  # 圖表外框
    #         ax.patch.set_alpha(0)   # 圖表繪製區塊

    #         ax.imshow(wordcloud, interpolation ='bilinear')
    #         ax.axis('off')
    #         # ax.set_title('展覽熱門關鍵字趨勢 (AI Tagging)', fontsize=16)

    #         # 6. 使用 Streamlit 顯示 Matplotlib 圖表
    #         st.pyplot(fig)
    #         plt.close(fig) # 關閉 Matplotlib 圖形，釋放記憶體

    #     except Exception as e:
    #         st.error(f'❌ 產生文字雲失敗: {e}')

    # 各session的頁面內容 ======================================================================
    # Session home
    def _home_session(self) -> None:
        # 頁面基礎資訊
        
        st.markdown(f'# **:orange[{self.topic}]**')    
        st.markdown('---')
        
        col_title, col_worldcloud = st.columns([3, 2]) # 讓搜尋欄位不佔滿整行
        with col_title:
            # with row_h, row_t = st.rows([3, 1])
            st.markdown(f'> 目前日期 &ensp; {dt.datetime.today().strftime('%Y-%m-%d')}')
            st.markdown(f'{self.sideprojectbrief}')

            # 用戶搜尋窗格
            st.markdown('##### **:red[想去哪裡看展?&emsp;&emsp;直接輸入找更快喔!]**')
            usr_input = st.text_input('搜尋展館', label_visibility = 'collapsed')
            filtered_venue_names = self._search_fuzzy_wildcard(usr_input, list(self.venue_image_urls.keys())) #
            
            # 整理 - 展覽的熱門關鍵字
            world_feq = []
            world_cloud_select = self.df_tags['hallname'].isin(filtered_venue_names) if filtered_venue_names else self.df_tags['hallname'].isin(list(self.venue_image_urls.keys()))
            df_tags_keywords = self.df_tags[world_cloud_select].copy(deep = True)
            df_tags_keywords['keywords'] = df_tags_keywords['keywords'].str.replace(r'[{}]', '', regex = True).str.split(',')
            for i in df_tags_keywords['keywords']:
                world_feq.extend(i)
            keyword_counts_series = pd.Series(world_feq, name = 'Tag').value_counts().reset_index(name = '出現次數').sort_values(by = '出現次數', ascending = False)
        

        # 20251206暫停功能 - 影響速度且移動到Tableau平台上面呈現就好
        # with col_worldcloud:
        #     st.markdown('### **:yellow[🔥 展覽關鍵字熱門趨勢(AI Tagging)]**')
        #     if not keyword_counts_series.empty:
        #         self._generate_wordcloud_plot(keyword_counts_series)
        #     else:
        #         st.caption('（尚無關鍵字資料可供分析）')

        st.markdown('---')

        if usr_input and filtered_venue_names != []:
            st.markdown('## 🏛️ 您可能要找的展館')
            st.info(f'**:yellow[🔥 全館前10大覽熱門關鍵字：]** {', '.join(keyword_counts_series['Tag'][:10].values)}')
            filtered_venue_info = {
                name : self.venue_image_urls[name] 
                for name in filtered_venue_names 
                if name in self.venue_image_urls
            } # 轉換成dict，為了要傳入版面呈現的函數中
            self._display_venue_grid(filtered_venue_info)

            st.markdown('---')
        else:
            if usr_input:
                st.markdown('### 找不到輸入的展覽館耶...請重新輸入，或是從下面圖片中找找看~')
                self._display_venue_grid(self.venue_image_urls)
            else:
                st.markdown('## 🏛️ 展覽場館一覽')
                st.info(f'**:yellow[🔥 雙北展覽前10大熱門關鍵字：]** {', '.join(keyword_counts_series['Tag'][:10].values)}')
                self._display_venue_grid(self.venue_image_urls)
                st.markdown('---')
                
                fut_venlist : List[str] = []
                for _, rows in self.df_future_venue.iterrows():
                    fut_venlist.append(rows['館名'])
                st.markdown(f'> :wrench: 持續新增中...&emsp;&emsp;{'、'.join(fut_venlist)}')
                
                
        
        
               
    
    # Session map_view
    def _map_view_session(self) -> None:
        # 返回按鈕
        if st.button('◀ 返回場館列表'):
            st.session_state['page_mode'] = 'home' # 切換回首頁
            st.rerun() # 重新執行應用程式以立即切換頁面
        
        # 頁面內容
        df_current_venue = self.df_exhibitions[self.df_exhibitions['展館名稱'] == st.session_state['selected']]
        st.markdown(f'# **:orange[{st.session_state['selected']}]**')
        st.markdown(f'> 目前日期 &ensp; {dt.datetime.today().strftime('%Y-%m-%d')}')
        st.markdown(f'**{self.venue_introduction.get(st.session_state['selected'])}**')
        st.markdown(f'官網連結 : {self.venue_urls.get(st.session_state['selected'])}')
        
        st.markdown('---')

        col_search, col_tag = st.columns([2, 3]) # 讓搜尋欄位不佔滿整行

        with col_search:
            st.markdown('##### **:red[有沒有要搜尋的展覽?&emsp;&emsp;直接輸入找更快喔!]**')
            usr_input = st.text_input('')
            checklist = self.df_exhibitions[self.df_exhibitions['展館名稱'] == st.session_state['selected']]['展覽名稱'].unique().tolist()
        st.markdown('---')


        filtered_exhibition_names = self._search_fuzzy_wildcard(usr_input, checklist) # 用戶可能再找的展覽清單
        # 整理 - 展覽的熱門關鍵字
        world_feq = []
        world_cloud_select = self.df_tags['title'].isin(filtered_exhibition_names) if filtered_exhibition_names else self.df_tags['title'].isin(checklist)
        df_tags_keywords = self.df_tags[world_cloud_select].copy(deep = True)
        df_tags_keywords['keywords'] = df_tags_keywords['keywords'].str.replace(r'[{}]', '', regex = True).str.split(',')
        for i in df_tags_keywords['keywords']:
            world_feq.extend(i)
        keyword_counts_series = pd.Series(world_feq, name = 'Tag').value_counts().reset_index(name = '出現次數').sort_values(by = '出現次數', ascending = False)
        hashtaglist = "`" + "` `".join(keyword_counts_series['Tag'].values) + "`"
        
        if usr_input and filtered_exhibition_names != []:
            df_display = df_current_venue[df_current_venue['展覽名稱'].isin(filtered_exhibition_names)]
            st.markdown(f' **:yellow[🔥 展覽關鍵字：]** ***{hashtaglist}***')
            self._display_venue_grid(df_display)

        else:

            if usr_input:
                st.markdown('### 找不到輸入的展覽館耶...請重新輸入，或是從下面圖片中找找看~')
                self._display_venue_grid(df_current_venue)
                
            else:
                st.markdown(f' **:yellow[🔥 展覽關鍵字：]** ***{hashtaglist}***')
                self._display_venue_grid(df_current_venue)

                


    # Session exhibition_view
    def _exhibition_view_session(self) -> None:
        select_ven = st.session_state['selected'] # 展覽資訊
        self._rec_system.record_click(e_name = select_ven, df = self.df_tags) # 記錄當下頁面中的標籤 > 後面要用來推薦的
        
        st.markdown(f'### 🗺️ **{select_ven}** 資訊')        
        
        
        st.markdown(f'{self.df_exhibitions[self.df_exhibitions['展覽名稱'] == select_ven]['網頁連結'].values[0]}')
        home_button, map_button, _ = st.columns([1, 1, 10])
        with map_button:
            if st.button('◀ 返回展覽列表'):
                st.session_state['page_mode'] = 'map_view' # 切換回展覽清單
                st.session_state['selected'] = self.df_exhibitions[self.df_exhibitions['展覽名稱'] == select_ven]['展館名稱'].unique().tolist()[0]
                st.rerun() # 重新執行應用程式以立即切換頁面
        with home_button:
            if st.button('◀ 返回場館列表'):
                st.session_state['page_mode'] = 'home' # 切換回展覽清單
                st.rerun()
            
            

        if not self.df_exhibitions.empty:
            select_df = self.df_exhibitions[self.df_exhibitions['展覽名稱'] == select_ven] # 篩出
            img_src = select_df['圖片連結'].values[0]
            st.markdown('---')
            # 整理 - 展覽的熱門關鍵字
            world_feq = []
            world_cloud_select = self.df_tags['title'].isin([select_ven])
            df_tags_keywords = self.df_tags[world_cloud_select].copy(deep = True)
            df_tags_keywords['keywords'] = df_tags_keywords['keywords'].str.replace(r'[{}]', '', regex = True).str.split(',')
            for i in df_tags_keywords['keywords']:
                world_feq.extend(i)
            keyword_counts_series = pd.Series(world_feq, name = 'Tag').value_counts().reset_index(name = '出現次數').sort_values(by = '出現次數', ascending = False)
            hashtaglist = "`" + "` `".join(keyword_counts_series['Tag'].values) + "`"
            st.markdown(f' **:yellow[🔥 展覽關鍵字：]** ***{hashtaglist}***')

            col_list, col_map = st.columns([2, 3]) # 3/5 寬度給地圖, 2/5 寬度給清單

            with col_list:
                reclist = self._rec_system.recomlist(df = self.df_tags) # 記錄當下頁面中的標籤 > 後面要用來推薦的
                rec_df = self.df_exhibitions[(self.df_exhibitions['展覽名稱'].isin([row for row in reclist if row not in select_df['展覽名稱'].unique().tolist()]))]

                infotext = []
                
                for loc in ['展覽地點', '展覽名稱', '開始日期', '結束日期', '參觀時間', '票價', '展覽介紹']:
                    infotext.append(f'**:yellow[{loc}]** : {select_df[loc].values[0]}')
                
                st.markdown('\n\n'.join(infotext))
                st.image(image = img_src, caption = f'**{select_df['展覽名稱'].values[0]}**')


            with col_map:
                
                st.markdown(f'### 周邊展覽地圖')
                # self._display_google_map(self.df_exhibitions, venue_name = select_df['展館名稱'].values[0], exhibition_name = select_ven ,map_height = 600)
                col_list_1, col_list_2 = st.columns([4, 1])
                with col_list_2:
                    lon = self.df_exhibitions[self.df_exhibitions['展覽名稱'] == select_ven]['經度'].values[0]
                    lat = self.df_exhibitions[self.df_exhibitions['展覽名稱'] == select_ven]['緯度'].values[0]
                    st.link_button(f'連線到google map', f'https://www.google.com/maps/search/?api=1&query={lat},{lon}')


            st.markdown('---')
            if reclist != None:
                st.markdown('##### :heart: :red[也許你會有興趣]')
                self._display_venue_grid(rec_df[:4])
    # 各session的頁面內容 ======================================================================            

   
    # Streamlit 應用程式主體 ====================================================================================
    def website_main(self):
        st.set_page_config(layout = 'wide', page_icon = '📊', page_title = self.config_ttile) # 設定 Streamlit 頁面標題和圖示，並設定為寬模式布局
        # 🎯 注入 CSS 以固定圖片高度
        st.markdown('''
            <style>
                /* 調整圖片大小 */
                .stImage img {
                    height: 250px !important; /* 設置您希望的固定高度，並使用 !important 提高權重 */
                    width: 100% !important; /* 確保寬度佔滿容器 */
                    object-fit: cover !important; /* 確保圖片不變形，會裁剪多餘部分，並使用 !important */
                    border-radius: 8px; /* 美化邊角 */
                }
                /* 為了美觀，可以讓圖片上方的容器 margin 消除一些 */
                div[data-testid="stImage"] {
                    margin-bottom: 0px; 
                }
                
            </style>
        ''', unsafe_allow_html = True)    

        if st.session_state['page_mode'] == 'home':
            self._home_session()
            
        elif st.session_state['page_mode'] == 'map_view':
            self._map_view_session()
            
        elif st.session_state['page_mode'] == 'exhibition_view':    
            self._exhibition_view_session()

        else:
            st.warning('資料庫連線失敗或沒有找到正在展出的展覽資料。請檢查錯誤訊息和連線字串。')

if __name__ == '__main__':
    load_dotenv() 
    app = streamlit_run_app()
    app.website_main()