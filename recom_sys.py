import streamlit as st
from collections import Counter
from typing import Optional, List, Dict, Set
import pandas as pd

# 使用 @st.cache_resource 確保這個類別實例在整個應用程式運行期間只會被創建一次
# 這樣它就能持久地保存 'tag_counts' 這個狀態
# 全域且不變的資源
@st.cache_resource
class RecommendationEngine: # 管理標籤點擊計數和推薦邏輯的核心引擎。
    def __init__(self):
        # 使用 Counter 來儲存標籤點擊次數，這是引擎的持久化狀態
        self._tag_counts = Counter()
        self.taglist : List = []
        with open('taglist.txt', 'r', encoding = 'utf-8') as f:
            for line in f:
                self.taglist.append(line)

    def record_click(self, e_name : str, df : pd.DataFrame) -> Dict[str, int] | None: # 記錄使用者點擊了哪個標籤。
        select_vename = df[df['title'].isin([e_name])]
        tags = select_vename['keywords'].str.replace(r'[{}]', '', regex = True).values[0].split(',')

        for tag in tags:
            self._tag_counts[tag] += 1 # 紀錄直接保存在實例中

    def _get_most_liked_tag(self) -> List | None: # 找出最高點級次數的標籤
        if not self._tag_counts: # 甚麼都沒有點
            return None
        
        # most_common(1) 返回 [(tag_name, count)]
        target_times = self._tag_counts.most_common(1)[0][1]
        most_common = [elem for elem, count in self._tag_counts.items() if count == target_times]
        return most_common

    def recomlist(self, df : pd.DataFrame) -> List[str] | None: # 返回前N大推薦標籤
        most_comlist = self._get_most_liked_tag()
        # st.markdown(most_comlist)
        if most_comlist:
            rec_list : List[str] = []
            
            for _, val in df.iterrows():
                exhibition_title = val['title']
                # st.dataframe(val[['title', 'keywords']])
                for j in val[['keywords']].str.replace(r'[{}]', '', regex = True).values[0].split(','):
                    if (j in most_comlist) and (exhibition_title not in rec_list):
                        rec_list.append(exhibition_title)
            return rec_list
        else:
            return None

# -------------------------------------------------------------
# 模組暴露的接口：獲取單例實例
# -------------------------------------------------------------

# 這個函式將成為外部調用的主要接口
def recommendation_engine() -> RecommendationEngine:
    # 由於類別本身已被 @st.cache_resource 快取，這個函式只會返回同一個實例
    return RecommendationEngine()