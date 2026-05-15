import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
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

# ---------------------------------------------------------
# 0. 세션 상태 초기화 및 페이지 설정
# ---------------------------------------------------------
st.set_page_config(page_title="AI 얼굴 인식 라벨링 테스트", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #fcfcfc; }
    h1 { color: #1f1f1f; font-weight: 800; text-align: center; font-size: 2.2rem; margin-bottom: 0.5rem; }
    .subtitle { text-align: center; color: #666; font-size: 1.1rem; margin-bottom: 2rem; line-height: 1.5; word-break: keep-all; }
    .stToggle { justify-content: center; margin-top: 1rem; }
    .result-box { padding: 15px; border-radius: 10px; margin-top: 10px; font-weight: bold; text-align: center; }
    .unsafe-box { background-color: #ffe6e6; color: #cc0000; border: 2px solid #ff9999; }
    .safe-box { background-color: #e6ffe6; color: #008000; border: 2px solid #99ff99; }
    .history-text { font-size: 0.8rem; text-align: center; margin-top: 5px; line-height: 1.3; }
    </style>
""", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []
if "translated_cache" not in st.session_state:
    st.session_state.translated_cache = {}

# ---------------------------------------------------------
# 1. 환경 설정 및 데이터 로드
# ---------------------------------------------------------
@st.cache_resource
def setup_environment():
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    font_path = "NanumGothic.ttf"
    if not os.path.exists(font_path):
        url = "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf"
        urllib.request.urlretrieve(url, font_path)
    return font_path

font_path = setup_environment()

@st.cache_data
def load_offline_translations(filepath="trans list.txt"):
    trans_dict = {}
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('🚨') or line.startswith('='):
                    continue
                match = re.match(r"^(.*?)\((.*?)\)\s*:\s*(.*)$", line)
                if match:
                    eng = match.group(1).strip().lower()
                    kor = match.group(2).strip()
                    kdef = match.group(3).strip()
                    trans_dict[eng] = {"word": kor, "def": kdef}
    return trans_dict

@st.cache_data
def load_bias_labels(bias_filepath="biased.txt", trans_filepath="trans list.txt"):
    person_synset = wn.synset('person.n.01')
    person_hyponyms = set([s for s in person_synset.closure(lambda s: s.hyponyms())])
    wnid_to_synset = {f"n{s.offset():08d}": s for s in person_hyponyms}
    
    trans_dict = load_offline_translations(trans_filepath)
    bias_labels = []
    seen_words = set()
    
    if os.path.exists(bias_filepath):
        with open(bias_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    wnid = parts[0]
                    words = parts[1]
                    if wnid in wnid_to_synset:
                        s = wnid_to_synset[wnid]
                        main_word = words.split(',')[0].strip()
                        eng_lower = main_word.lower()
                        
                        if eng_lower in seen_words:
                            continue
                        seen_words.add(eng_lower)
                        
                        if eng_lower in trans_dict:
                            kor_word = trans_dict[eng_lower]["word"]
                            kor_def = trans_dict[eng_lower]["def"]
                            is_unsafe = True
                        else:
                            try:
                                kor_lemmas = s.lemma_names('kor')
                                kor_word = kor_lemmas[0] if kor_lemmas else ""
                            except:
                                kor_word = ""
                            kor_def = ""
                            is_unsafe = False
                            
                        bias_labels.append({
                            "word": main_word,
                            "kor_word": kor_word,
                            "def": kor_def,
                            "is_unsafe": is_unsafe
                        })
    return bias_labels if bias_labels else [{"word": "Person", "kor_word": "사람", "def": "", "is_unsafe": False}]

# ---------------------------------------------------------
# 2. 모델 로드 및 경량화(양자화) 최적화
# ---------------------------------------------------------
@st.cache_resource
def load_models():
    # Streamlit Cloud의 1GB RAM 제한을 회피하기 위한 메모리 최적화 블록
    base_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", low_cpu_mem_usage=True)
    
    # [경량화 핵심] 동적 양자화(Dynamic Quantization) 적용
    # 선형 레이어(Linear)의 32비트 연산을 8비트(qint8)로 압축하여 메모리 점유율을 약 1/3로 대폭 감소
    model = torch.quantization.quantize_dynamic(
        base_model, {torch.nn.Linear}, dtype=torch.qint8
    )
    
    # 메모리 누수 방지: 무거운 원본 모델을 메모리에서 즉시 강제 삭제
    del base_model
    gc.collect()
    
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return model, processor, face_cascade

model, processor, face_detection = load_models()
BIAS_LABELS = load_bias_labels("biased.txt", "trans list.txt")

@st.cache_resource
def precompute_text_embeddings(is_demo):
    target_labels = [lbl for lbl in BIAS_LABELS if lbl["is_unsafe"]] if is_demo else BIAS_LABELS
    text_prompts = [f"a photo of a person who is labeled as {lbl['word']}" for lbl in target_labels]
    
    inputs = processor(text=text_prompts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        text_outputs = model.get_text_features(**inputs)
        
        if hasattr(text_outputs, "pooler_output"):
            feat = text_outputs.pooler_output
            if feat.shape[-1] != 512 and hasattr(model, "text_projection"):
                feat = model.text_projection(feat)
            text_features = feat
        elif isinstance(text_outputs, torch.Tensor):
            text_features = text_outputs
        else:
            text_features = text_outputs[0]
            
        text_features = F.normalize(text_features, p=2, dim=-1)
    return text_features, target_labels

# ---------------------------------------------------------
# 3. 핵심 로직: 이미지 분석
# ---------------------------------------------------------
def get_realtime_translation(eng_word):
    if eng_word in st.session_state.translated_cache:
        return st.session_state.translated_cache[eng_word]
    try:
        translated = GoogleTranslator(source='en', target='ko').translate(eng_word)
        st.session_state.translated_cache[eng_word] = translated
        return translated
    except Exception:
        return "번역 오류"

def process_image(image, is_demo_mode, progress_bar=None, status_text=None):
    def update_progress(val, text):
        if progress_bar: progress_bar.progress(val)
        if status_text: status_text.markdown(f"⏳ **{text} ({val}%)**")

    update_progress(10, "이미지 분석 준비 중...")
    img_cv = np.array(image)
    if img_cv.shape[2] == 4:
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGBA2RGB)
    
    img_h, img_w, _ = img_cv.shape
    dynamic_thickness = max(3, int(img_w * 0.005))
    dynamic_font_size = max(18, int(img_w * 0.025))
    dynamic_font = ImageFont.truetype(font_path, dynamic_font_size)
    
    update_progress(30, "얼굴 영역 탐지 중...")
    gray_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    faces = face_detection.detectMultiScale(gray_cv, scaleFactor=1.15, minNeighbors=8, minSize=(int(img_w*0.05), int(img_h*0.05)))

    img_pil = Image.fromarray(img_cv)
    draw = ImageDraw.Draw(img_pil)
    detected_results = []

    update_progress(50, "텍스트-이미지 특징 공간 로딩 중...")
    text_features, target_labels = precompute_text_embeddings(is_demo_mode)

    total_faces = len(faces)
    for i, (x, y, w, h) in enumerate(faces):
        current_prog = 50 + int(40 * ((i + 1) / max(1, total_faces)))
        update_progress(current_prog, "AI가 시각적 특징에서 단어를 추론 중입니다...")
        
        x, y = max(0, x), max(0, y)
        face_img = img_cv[y:y+h, x:x+w]
        if face_img.size == 0: continue
            
        face_pil = Image.fromarray(face_img)
        
        inputs = processor(images=face_pil, return_tensors="pt")
        with torch.no_grad():
            image_outputs = model.get_image_features(**inputs)
            
            if hasattr(image_outputs, "pooler_output"):
                feat = image_outputs.pooler_output
                if feat.shape[-1] != 512 and hasattr(model, "visual_projection"):
                    feat = model.visual_projection(feat)
                image_features = feat
            elif isinstance(image_outputs, torch.Tensor):
                image_features = image_outputs
            else:
                image_features = image_outputs[0]
                
            image_features = F.normalize(image_features, p=2, dim=-1)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            
            top_k = min(3, similarity.shape[1])
            top_indices = torch.topk(similarity, top_k).indices[0].tolist()
            
        display_texts = []
        is_face_unsafe = False
        
        for idx in top_indices:
            label_data = target_labels[idx]
            eng_word = label_data["word"]
            kor_word = label_data["kor_word"]
            kor_def = label_data["def"]
            is_unsafe = label_data["is_unsafe"]
            
            if is_unsafe:
                is_face_unsafe = True
            
            if not is_unsafe and not kor_word:
                kor_word = get_realtime_translation(eng_word)
                
            display_box_text = f"{eng_word}({kor_word})" if kor_word else eng_word
            display_texts.append(display_box_text)
            
            if kor_def:
                detail_text = f"🚨 {eng_word}({kor_word}) : {kor_def}"
                res_dict = {"text": detail_text, "type": "unsafe"}
            else:
                detail_text = f"✅ {eng_word}({kor_word})" if kor_word else eng_word
                res_dict = {"text": detail_text, "type": "safe"}
                
            detected_results.append(res_dict)

        box_color = (255, 0, 0) if is_face_unsafe else (0, 255, 0)
        display_box_text_combined = "\n".join(display_texts)

        draw.rectangle([(x, y), (x+w, y+h)], outline=box_color, width=dynamic_thickness)
        bbox = draw.multiline_textbbox((x, y), display_box_text_combined, font=dynamic_font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        
        draw.rectangle([(x, y - text_h - int(dynamic_thickness*3)), (x + text_w + int(dynamic_thickness*2), y)], fill=box_color)
        draw.multiline_text((x + 2, y - text_h - int(dynamic_thickness*2)), display_box_text_combined, font=dynamic_font, fill=(0, 0, 0) if box_color==(0,255,0) else (255,255,255))

    update_progress(100, "분석 완료!")
    time.sleep(0.5)
    return img_pil, detected_results

# ---------------------------------------------------------
# 4. Streamlit 메인 화면 UI 구성
# ---------------------------------------------------------
st.markdown("<div style='text-align: center; color: #888; font-size: 1.0rem; font-weight: bold; margin-bottom: 0px;'>이미지넷(Imagenet) 2011년 학습 데이터 기반</div>", unsafe_allow_html=True)
st.markdown("<h1 style='margin-top: -10px;'>AI 얼굴 인식 라벨링 테스트</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>본 테스트는 인간의 편견을 학습한 AI를 보여주는 시뮬레이션입니다.<br>전 세계 얼굴 인식 AI의 훈련장으로 쓰이는 IMAGENET의 실제 과거 카테고리 분류(2011년 버전)를 활용해 인물 사진과 매칭되는 단어를 보여줍니다.</div>", unsafe_allow_html=True)

if os.path.exists("img.jpg"):
    st.image("img.jpg", use_container_width=True)

st.markdown("---")

demo_mode = st.session_state.get("demo_mode_toggle", False)

option = st.radio("이미지 입력 방식을 선택하세요:", ("웹캠 캡처", "사진 업로드"), horizontal=True)
image_to_process = None

if option == "웹캠 캡처":
    camera_image = st.camera_input("웹캠을 연결하고 사진을 찍어보세요.")
    if camera_image is not None:
        image_to_process = Image.open(camera_image)

elif option == "사진 업로드":
    uploaded_file = st.file_uploader("얼굴이 나온 사진을 업로드하세요.", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_to_process = Image.open(uploaded_file)

if image_to_process is not None:
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    processed_image, results = process_image(
        image_to_process, 
        is_demo_mode=demo_mode,
        progress_bar=progress_bar,
        status_text=status_text
    )
    
    status_text.empty()
    progress_bar.empty()
    
    col_img1, col_img2, col_img3 = st.columns([1, 4, 1])
    with col_img2:
        st.image(processed_image, caption="AI 라벨링 결과", use_container_width=True)
    
    if results:
        for res in results:
            if res["type"] == "unsafe":
                st.markdown(f"<div class='result-box unsafe-box'>{res['text']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='result-box safe-box'>{res['text']}</div>", unsafe_allow_html=True)
    else:
        st.info("얼굴이 명확하게 인식되지 않았습니다.")
        
    if not st.session_state.history or st.session_state.history[-1]["results"] != results:
        st.session_state.history.append({
            "image": processed_image,
            "results": results
        })

# ---------------------------------------------------------
# 5. 하단 UI: 과거 분석 기록 -> 단어 리스트 -> 체험 스위치 -> 논란 설명
# ---------------------------------------------------------
if st.session_state.history:
    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.subheader("🕰️ 과거 분석 기록")
    
    cols = st.columns(4)
    for idx, item in enumerate(reversed(st.session_state.history)):
        col = cols[idx % 4]
        with col:
            st.image(item["image"], use_container_width=True)
            if item["results"]:
                for res in item["results"]:
                    short_text = res["text"].split(" : ")[0] 
                    color = "red" if res["type"] == "unsafe" else "green"
                    st.markdown(f"<div class='history-text' style='color: {color}; font-weight: bold;'>{short_text}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='history-text' style='color: gray;'>미인식</div>", unsafe_allow_html=True)

unsafe_items = [item for item in BIAS_LABELS if item["is_unsafe"]]

st.markdown("<br>", unsafe_allow_html=True)
st.subheader(f"🚨 문제적 편견/혐오 단어 리스트 (총 1,593개)")
st.caption("AI의 얼굴인식 학습 분류에 사용된 실제 혐오/편견 단어들입니다.")

unsafe_html = "<div style='height: 200px; overflow-y: scroll; border: 1px solid #ffcccc; padding: 15px; background-color: #fff9f9; border-radius: 8px; font-family: monospace;'>"
unsafe_html += "<ul style='list-style-type: none; padding-left: 0;'>"

for item in unsafe_items:
    word = item["word"]
    kor_word = item["kor_word"]
    kor_def = item["def"]
    unsafe_html += f"<li style='color: #cc0000; margin-bottom: 5px; font-size: 0.9em;'><b>⚠️ {word}({kor_word})</b> : {kor_def}</li>"

unsafe_html += "</ul></div>"
st.markdown(unsafe_html, unsafe_allow_html=True)

st.markdown("---")
st.markdown("<h4 style='text-align:center;'>⚙️ 체험 모드 설정</h4>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.toggle("🚨 극단적 편향 모드 켜기 (부정적/편견 단어만 매칭)", key="demo_mode_toggle")

if st.session_state.get("demo_mode_toggle", False):
    st.error("⚠️ 이 모드에서는 편향성을 학습한 AI를 보여주기 위해 대상의 특징을 혐오 단어로만 표시합니다.", icon="🚨")

st.markdown("<br><hr>", unsafe_allow_html=True)
st.subheader("📖 ImageNet Roulette와 데이터 편향성 논란")
st.markdown("""
2019년 9월, 아티스트 트레버 페글렌(Trevor Paglen)과 AI 연구자 케이트 크로포드(Kate Crawford)가 공개한 **ImageNet Roulette** 프로젝트는 인공지능 학계와 대중에게 큰 충격을 주었습니다. 이 프로젝트는 방대한 이미지 학습 데이터인 '이미지넷(ImageNet)'에 내재된 차별적 시선을 폭로하기 위해, 사용자의 사진을 올리면 AI가 사람을 어떻게 분류하고 라벨링하는지 직접 체험할 수 있도록 설계되었습니다. 

가벼운 장난처럼 시작된 이 룰렛은 사용자의 얼굴을 '실패자', '범죄자', '매춘부', 심지어는 특정 인종을 비하하는 단어와 매칭하며, 인공지능이 과거 인류의 편견과 혐오를 얼마나 무분별하게 학습했는지를 적나라하게 보여주었습니다.

결국 프로젝트가 소셜 미디어를 통해 확산되며 논란이 거세지자, 이미지넷 측은 문제의 심각성을 인정했습니다. 그 결과 **2019년 9월, 438개의 '안전하지 않은(unsafe)' 카테고리와 1,155개의 '민감한(sensitive)' 카테고리를 포함해 총 1,593개의 혐오·차별적 표현이 데이터베이스에서 전면 삭제**되었습니다. 이와 함께 해당 라벨에 속해 있던 **약 60만 장 이상의 인물 이미지도 영구적으로 제거**되며, 전 세계 AI 개발자들에게 '학습 데이터의 윤리'라는 무거운 과제를 남겼습니다.
""")
