import streamlit as st
import streamlit.components.v1 as components
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont, ImageOps
from transformers import CLIPProcessor, CLIPModel
from deep_translator import GoogleTranslator
import nltk
from nltk.corpus import wordnet as wn
import os
import urllib.request
import re
import random
import time
import gc
import io

# ---------------------------------------------------------
# 0. 세션 상태 초기화 및 페이지 설정
# ---------------------------------------------------------

st.set_page_config(page_title="AI 얼굴 인식 라벨링 테스트", layout="centered")

# 카카오톡 인앱 브라우저 감지 및 외부 브라우저 자동 호출 로직
components.html(
    """
    <script>
    var ua = navigator.userAgent.toLowerCase();
    if (ua.indexOf('kakaotalk') > -1) {
        // iframe 내부에서 실행되므로 부모 창의 URL을 참조(CORS 우회)
        var target_url = document.referrer;
        if (!target_url) { target_url = window.location.href; }
        
        var scheme_url = 'kakaotalk://web/openExternal?url=' + encodeURIComponent(target_url);
        
        try {
            window.parent.location.href = scheme_url;
        } catch (e) {
            location.href = scheme_url;
        }
    }
    </script>
    """,
    width=0, height=0
)

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

* { box-sizing: border-box; }

/* 1. 전역 폰트 및 한국어 어절 단위 줄바꿈 설정 */
html, body, [class*="css"] { 
    font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    word-break: keep-all; 
    overflow-wrap: break-word; 
}

.main { background-color: #f8f7f4; }
.block-container { padding-top: 2.5rem; padding-bottom: 3rem; max-width: 780px; }

/* 상단 태그 */
.eyebrow { text-align: center; font-family: 'DM Mono', monospace; font-size: 0.72rem; letter-spacing: 0.12em; color: #999; text-transform: uppercase; margin-bottom: 0.6rem; }

/* 메인 타이틀 */
h1 { display: block; width: 100%; text-align: center !important; font-size: 2rem !important; font-weight: 600 !important; color: #111 !important; letter-spacing: -0.03em; line-height: 1.3; margin-bottom: 1rem !important; word-break: keep-all; }

/* 서브타이틀 */
.subtitle { text-align: center; color: #666; font-size: 0.93rem; line-height: 1.6; max-width: 560px; margin: 0 auto 2rem auto; word-break: keep-all; }

/* 구분선 */
.divider { border: none; border-top: 1px solid #e0ddd8; margin: 2rem 0; }

/* 결과 박스 */
.result-box { padding: 12px 16px; border-radius: 6px; margin-top: 8px; font-size: 0.9rem; font-weight: 500; text-align: center; letter-spacing: -0.01em; word-break: keep-all; }
.unsafe-box { background-color: #fff0ef; color: #c0392b; border: 1px solid #f5c6c3; }
.safe-box { background-color: #f0f7f1; color: #2e7d45; border: 1px solid #b8dfc1; }

/* 인물 헤더 */
.person-header { text-align: center; font-weight: 600; font-size: 0.9rem; margin-top: 1.4rem; color: #333; letter-spacing: -0.01em; word-break: keep-all; }

/* 범례 박스 */
.legend-box { text-align: center; font-size: 0.82rem; color: #777; background-color: #f0ede8; padding: 10px 16px; border-radius: 6px; margin-top: 1.4rem; line-height: 1.6; word-break: keep-all; }

/* 히스토리 텍스트 */
.history-text { font-size: 0.75rem; text-align: center; margin-top: 4px; line-height: 1.4; font-family: 'DM Mono', monospace; word-break: keep-all; }

/* 에러 박스 (업로드 실패) */
.upload-error { background-color: #fff8ed; border: 1px solid #f5d9a3; border-radius: 6px; padding: 12px 16px; font-size: 0.88rem; color: #a05c00; margin-top: 10px; word-break: keep-all; }

/* 섹션 헤더 및 앵커 링크(쇠사슬 아이콘) 숨김 처리 */
h2, h3, h4 { font-weight: 600 !important; letter-spacing: -0.02em !important; color: #111 !important; word-break: keep-all; text-align: center; } 
a.header-anchor { display: none !important; }

/* Streamlit 기본 요소 오버라이드 */
.stRadio > label { font-size: 0.88rem; color: #555; word-break: keep-all; }
.stRadio [role="radiogroup"] { gap: 8px; }
.stMarkdown p { word-break: keep-all; overflow-wrap: break-word; }

/* 토글 UI 강제 정렬 및 텍스트 쪼개짐 방지 */
.stToggle { display: flex; justify-content: center; align-items: center; width: 100%; margin: 0 auto; }
.stToggle label { display: flex !important; justify-content: center !important; align-items: center !important; }
.stToggle label, 
.stToggle div, 
.stToggle p, 
.stToggle span { white-space: nowrap !important; word-break: keep-all !important; }

/* 진행 바 텍스트 */
.progress-label { font-family: 'DM Mono', monospace; font-size: 0.78rem; color: #888; text-align: center; margin-top: 4px; word-break: keep-all; }

/* 스크롤 단어 리스트 */
.word-list-container { height: 200px; overflow-y: auto; border: 1px solid #e0ddd8; padding: 14px 18px; background-color: #fff; border-radius: 8px; font-family: 'DM Mono', monospace; }

/* 모드 섹션 */
.mode-section { background-color: #fff; border: 1px solid #e0ddd8; border-radius: 10px; padding: 1.2rem 1.6rem; text-align: center; word-break: keep-all; }

/* 본문 텍스트 */
.article-text { font-size: 0.92rem; line-height: 1.85; color: #444; word-break: keep-all; }
.article-text b { color: #111; font-weight: 600; }

/* 2. 모바일 및 태블릿 대응 반응형 디자인 */
@media screen and (max-width: 768px) {
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; padding-left: 1.2rem; padding-right: 1.2rem; }
    h1 { font-size: 1.5rem !important; }
    .subtitle { font-size: 0.85rem; padding: 0 5px; }
    .mode-section { padding: 1rem; }
    .word-list-container { padding: 10px 12px; }
    .stToggle p { font-size: 0.85rem !important; }
}

@media screen and (max-width: 480
