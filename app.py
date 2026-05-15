import streamlit as st
import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import nltk
from nltk.corpus import wordnet as wn
import os
import urllib.request
import hashlib
import re

# ---------------------------------------------------------
# 0. 세션 상태 초기화
# ---------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------------------------------------
# 1. 환경 설정 및 오프라인 번역 파일(trans list.txt) 로드
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
                # 빈 줄이나 구분선, 제목 등은 건너뜀
                if not line or line.startswith('🚨') or line.startswith('='):
                    continue
                
                # 정규식을 이용해 "영어(한국어): 뜻풀이" 패턴을 정확히 파싱
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
    
    # 준비해주신 번역 파일 로드
    trans_dict = load_offline_translations(trans_filepath)
    
    bias_labels = []
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
                        
                        # trans list.txt에 단어가 있으면 위험/편견 단어로 지정하고 뜻풀이 연동
                        if eng_lower in trans_dict:
                            kor_word = trans_dict[eng_lower]["word"]
                            kor_def = trans_dict[eng_lower]["def"]
                            is_unsafe = True
                        else:
                            # 번역 리스트에 없으면 일반(안전) 단어로 간주
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
# 2. 모델 로드 (★ 이 부분만 resnet18에서 mobilenet_v2로 교체됨 ★)
# ---------------------------------------------------------
@st.cache_resource
def load_models():
    # 서버 메모리 초과를 막기 위해 초경량 mobilenet_v2 적용 (원작의 시각적 편향성은 그대로 유지됨)
    model = models.mobilenet_v2(pretrained=True)
    model.eval()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return model, face_cascade

model, face_detection = load_models()
BIAS_LABELS = load_bias_labels("biased.txt", "trans list.txt")

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---------------------------------------------------------
# 3. 핵심 로직: 얼굴 인식 및 오프라인 매핑 (로직 유지)
# ---------------------------------------------------------
def process_image(image):
    img_cv = np.array(image)
    if img_cv.shape[2] == 4:
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGBA2RGB)
    
    img_h, img_w, _ = img_cv.shape
    dynamic_thickness = max(3, int(img_w * 0.005))
    dynamic_font_size = max(18, int(img_w * 0.025))
    dynamic_font = ImageFont.truetype(font_path, dynamic_font_size)
    
    gray_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    faces = face_detection.detectMultiScale(gray_cv, scaleFactor=1.15, minNeighbors=8, minSize=(int(img_w*0.05), int(img_h*0.05)))

    img_pil = Image.fromarray(img_cv)
    draw = ImageDraw.Draw(img_pil)
    detected_results = []

    for (x, y, w, h) in faces:
        x, y = max(0, x), max(0, y)
        face_img = img_cv[y:y+h, x:x+w]
        if face_img.size == 0: continue
            
        face_pil = Image.fromarray(face_img)
        input_tensor = preprocess(face_pil).unsqueeze(0)
        
        with torch.no_grad():
            features = model(input_tensor)
            feature_str = str(np.round(features.numpy().flatten(), 2).tolist())
            hash_val = int(hashlib.md5(feature_str.encode()).hexdigest(), 16)
            label_index = hash_val % len(BIAS_LABELS)
            
            label_data = BIAS_LABELS[label_index]
            eng_word = label_data["word"]
            kor_word = label_data["kor_word"]
            kor_def = label_data["def"]
            
            # 텍스트 출력 포맷 깔끔하게 적용 (뜻풀이가 있을 때만 출력)
            display_box_text = f"{eng_word}({kor_word})" if kor_word else eng_word
            
            if kor_def:
                detail_text = f"{eng_word}({kor_word}) : {kor_def}"
            else:
                detail_text = f"{eng_word}({kor_word})" if kor_word else eng_word
                
            detected_results.append(detail_text)

        # 사진 위에 바운딩 박스와 텍스트 합성
        draw.rectangle([(x, y), (x+w, y+h)], outline=(0, 255, 0), width=dynamic_thickness)
        bbox = draw.textbbox((x, y), display_box_text, font=dynamic_font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        
        draw.rectangle([(x, y - text_h - int(dynamic_thickness*3)), (x + text_w + int(dynamic_thickness*2), y)], fill=(0, 255, 0))
        draw.text((x + 2, y - text_h - int(dynamic_thickness*2)), display_box_text, font=dynamic_font, fill=(0, 0, 0))

    return img_pil, detected_results

# ---------------------------------------------------------
# 4. Streamlit UI
# ---------------------------------------------------------
st.set_page_config(page_title="ImageNet 2011 Training Data-based Face Recognition", layout="centered")

st.title("이미지넷(Imagenet) 2011년 학습 데이터 기반 얼굴 인식")
st.caption("오염된 학습 데이터셋으로 AI가 인간 얼굴을 자의적으로 판단하는 구조 시각화")

st.markdown("---")

option = st.radio("이미지 입력 방식을 선택하세요:", ("웹캠 캡처", "사진 업로드"))
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
    with st.spinner("AI가 특징을 분석하여 오프라인 매핑 중입니다..."):
        processed_image, results = process_image(image_to_process)
        st.image(processed_image, caption="분석 결과", width="stretch")
        
        if results:
            for res in results:
                st.info(res)
        else:
            st.info("얼굴이 명확하게 인식되지 않았습니다.")
            
        if not st.session_state.history or st.session_state.history[-1]["results"] != results:
            st.session_state.history.append({
                "image": processed_image,
                "results": results
            })

# ---------------------------------------------------------
# 과거 기록 썸네일 섹션
# ---------------------------------------------------------
if st.session_state.history:
    st.markdown("---")
    st.subheader("🕰️ 과거 분석 썸네일 기록")
    
    cols = st.columns(4)
    for idx, item in enumerate(reversed(st.session_state.history)):
        col = cols[idx % 4]
        with col:
            st.image(item["image"], width="stretch")
            if item["results"]:
                for res in item["results"]:
                    st.markdown(f"<div style='font-size: 0.8em; line-height: 1.2;'>{res}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='font-size: 0.8em;'>얼굴 미인식</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# 1. 문제적 편견 혐오 단어 별도 리스트 (스크롤)
# ---------------------------------------------------------
unsafe_items = [item for item in BIAS_LABELS if item["is_unsafe"]]

st.markdown("---")
st.subheader(f"🚨 문제적 편견/혐오 단어 리스트 (총 {len(unsafe_items)}개)")
st.caption("`trans list.txt` 데이터를 기반으로 검출된 실제 삭제된 혐오/편견 단어들입니다.")

unsafe_html = "<div style='height: 250px; overflow-y: scroll; border: 2px solid red; padding: 15px; background-color: #fff0f0; border-radius: 8px; font-family: monospace;'>"
unsafe_html += "<ul style='list-style-type: none; padding-left: 0;'>"

for item in unsafe_items:
    word = item["word"]
    kor_word = item["kor_word"]
    kor_def = item["def"]
    
    unsafe_html += f"<li style='color: red; margin-bottom: 5px;'><b>⚠️ {word}({kor_word})</b> : {kor_def}</li>"

unsafe_html += "</ul></div>"
st.markdown(unsafe_html, unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. 전체 단어 리스트 (오른쪽 스크롤 바)
# ---------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
st.subheader(f"📋 ImageNet 2011 '사람' 카테고리 전체 목록 (총 {len(BIAS_LABELS)}개)")
st.caption("당시 AI가 사람의 얼굴을 분류하는 데 사용했던 전체 단어 풀(Pool)입니다.")

full_html = "<div style='height: 300px; overflow-y: scroll; border: 1px solid #ddd; padding: 15px; background-color: #f9f9f9; border-radius: 8px; font-family: monospace;'>"
full_html += "<ul style='list-style-type: none; padding-left: 0;'>"

for item in BIAS_LABELS:
    word = item["word"]
    if item["is_unsafe"]:
        kor_word = item["kor_word"]
        full_html += f"<li style='color: red; font-weight: bold; margin-bottom: 3px;'>⚠️ {word}({kor_word})</li>"
    else:
        full_html += f"<li style='color: #444; margin-bottom: 3px;'>{word}</li>"

full_html += "</ul></div>"
st.markdown(full_html, unsafe_allow_html=True)
