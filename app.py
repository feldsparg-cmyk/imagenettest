import streamlit as st
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

@media screen and (max-width: 480px) {
   h1 { font-size: 1.3rem !important; }
   .result-box { font-size: 0.85rem; padding: 10px 12px; }
   .article-text { font-size: 0.85rem; line-height: 1.6; }
   .stToggle p { font-size: 0.75rem !important; }
}
</style>""", unsafe_allow_html=True)

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
else:
match_no_def = re.match(r"^(.*?)\((.*?)\)", line)
if match_no_def:
eng = match_no_def.group(1).strip().lower()
kor = match_no_def.group(2).strip()
trans_dict[eng] = {"word": kor, "def": ""}
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
# 2. 모델 로드
# ---------------------------------------------------------

@st.cache_resource
def load_models():
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
return model, processor, face_cascade

model, processor, face_detection = load_models()
BIAS_LABELS = load_bias_labels("biased.txt", "trans list.txt")

# ---------------------------------------------------------
# 3. 단어 임베딩 전역 캐시 처리 
# ---------------------------------------------------------
@st.cache_resource(show_spinner="AI 단어 사전 모델을 초기화 중입니다 (최초 1회만 소요됩니다)...")
def get_text_embeddings(is_demo):
target_labels = [lbl for lbl in BIAS_LABELS if lbl["is_unsafe"]] if is_demo else BIAS_LABELS
text_prompts = [f"a photo of a person who is labeled as {lbl['word']}" for lbl in target_labels]

all_text_features = []
batch_size = 128 

for i in range(0, len(text_prompts), batch_size):
batch_prompts = text_prompts[i:i+batch_size]
inputs = processor(text=batch_prompts, return_tensors="pt", padding=True, truncation=True)
with torch.no_grad():
text_outputs = model.get_text_features(**inputs)
if hasattr(text_outputs, "pooler_output"):
feat = text_outputs.pooler_output
if feat.shape[-1] != 512 and hasattr(model, "text_projection"):
feat = model.text_projection(feat)
elif isinstance(text_outputs, torch.Tensor):
feat = text_outputs
else:
feat = text_outputs[0]
feat = F.normalize(feat, p=2, dim=-1)
all_text_features.append(feat)

del inputs, text_outputs
gc.collect()

text_features = torch.cat(all_text_features, dim=0)
return text_features, target_labels

# ---------------------------------------------------------
# 4. 강력한 이미지 통합 전처리 함수 (구형 JPG 오류 완벽 대응 - OpenCV 선처리)
# 4. 강력한 이미지 통합 전처리 함수 (안정성 최우선 로직)
# ---------------------------------------------------------

def load_and_prep_image(file_or_cam):
img = None
    raw_bytes = None

    
try:
        # 파일 포인터를 바이트로 읽기 (버그 방지를 위해 포인터 강제 초기화)
        # 1. 파일에서 바이트 데이터를 안전하게 추출 (포인터 문제 완벽 차단)
if hasattr(file_or_cam, 'read'):
file_or_cam.seek(0)
raw_bytes = file_or_cam.read()
            if not raw_bytes or len(raw_bytes) < 8:
                return None, "파일이 비어있거나 손상되었습니다."
            stream = io.BytesIO(raw_bytes)
        elif hasattr(file_or_cam, 'getvalue'):
            raw_bytes = file_or_cam.getvalue()
else:
            stream = file_or_cam
            if hasattr(stream, 'getvalue'):
                raw_bytes = stream.getvalue()

        # [핵심 수정] 무조건 강력한 OpenCV로 먼저 디코딩 시도
        cv_success = False
        if raw_bytes:
            try:
                np_arr = np.frombuffer(raw_bytes, np.uint8)
                cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if cv_img is not None:
                    # OpenCV 성공 시 BGR을 RGB로 변환하여 완전한 PIL Image 생성
                    cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(cv_img_rgb)
                    cv_success = True
            except Exception:
                pass

        # OpenCV가 실패한 경우에만 PIL로 2차 시도
        if not cv_success:
            from PIL import ImageFile
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            try:
                img = Image.open(stream)
                img.load()
            except Exception as e:
                err_lower = str(e).lower()
                if "cannot identify" in err_lower or "no decoder" in err_lower:
                    return None, "지원하지 않는 파일 형식입니다. (JPG, PNG, WebP, BMP, TIFF 권장)"
                elif "truncated" in err_lower or "premature" in err_lower:
                    return None, "파일이 불완전하게 저장되어 있습니다. 다시 저장 후 업로드해 보세요."
                else:
                    return None, f"이미지를 열 수 없습니다: {str(e)}"
            return None, "지원하지 않는 입력 형식입니다."

        if not raw_bytes or len(raw_bytes) < 8:
            return None, "파일이 비어있거나 손상되었습니다."

        # 독립적인 바이트 스트림 생성
        stream = io.BytesIO(raw_bytes)
        
        # 2. PIL ImageFile 설정: 잘린 이미지 강제 허용
        from PIL import ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        # 3. PIL을 이용해 가장 표준적인 방법으로 이미지 로드
        try:
            img = Image.open(stream)
            img.load()  # 실제 데이터 메모리 적재 시도
        except Exception as e:
            return None, f"이미지를 해독할 수 없습니다. 형식이 잘못되었거나 심각하게 손상된 파일입니다: {str(e)}"

        # EXIF 회전 보정 (iPhone, Android 사진 대응)
        # 4. EXIF 회전 보정 (실패하더라도 이미지가 날아가지 않도록 안전하게 처리)
try:
            img = ImageOps.exif_transpose(img)
            # ImageOps.exif_transpose는 원본 객체를 변경하므로 실패 시 원본 유지가 중요함
            transposed_img = ImageOps.exif_transpose(img)
            if transposed_img is not None:
                img = transposed_img
except Exception:
            pass

        # 색 공간 통합 변환 (모든 모드 → RGB)
        if img.mode == 'RGBA':
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background
        elif img.mode == 'CMYK':
            img = img.convert('RGB')
        elif img.mode == 'P':
            img = img.convert('RGBA')
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background
        elif img.mode == 'LA':
            img = img.convert('RGBA')
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background
        elif img.mode == 'L':
            img = img.convert('RGB')
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        # 최소 크기 검사
            pass # EXIF 정보가 없거나 오류가 나면 무시하고 원본 img 객체 그대로 사용

        # 5. 색 공간 통합 변환 (모든 모드 → RGB)
        try:
            if img.mode == 'RGBA':
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background
            elif img.mode == 'CMYK':
                img = img.convert('RGB')
            elif img.mode in ('P', 'LA'):
                img = img.convert('RGBA')
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
        except Exception as e:
             return None, f"이미지 색상 변환 중 오류가 발생했습니다: {str(e)}"

        # 6. 최소/최대 크기 검사 및 조정
w, h = img.size
if w < 50 or h < 50:
return None, f"이미지가 너무 작습니다 ({w}×{h}px). 더 큰 이미지를 업로드해 주세요."

        # 최대 크기 제한 (메모리 절약)
max_size = 1200
if max(w, h) > max_size:
if w > h:
new_w, new_h = max_size, int(h * (max_size / w))
else:
new_h, new_w = max_size, int(w * (max_size / h))
img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

return img, None

except MemoryError:
return None, "이미지 파일이 너무 큽니다. 더 작은 파일을 업로드해 주세요."
except Exception as e:
return None, f"알 수 없는 오류가 발생했습니다: {str(e)}"

# ---------------------------------------------------------
# 5. 이미지 분석 코어
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
if status_text: status_text.markdown(f"<div class='progress-label'>{text} — {val}%</div>", unsafe_allow_html=True)

update_progress(10, "이미지 분석 준비 중")

img_cv = np.array(image)
img_h, img_w, _ = img_cv.shape

dynamic_thickness = max(2, int(img_w * 0.005))
dynamic_font_size = max(16, int(img_w * 0.025))
dynamic_font = ImageFont.truetype(font_path, dynamic_font_size)

update_progress(30, "얼굴 영역 탐지 중")

gray_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
faces = face_detection.detectMultiScale(
gray_cv,
scaleFactor=1.05,
minNeighbors=3,
minSize=(50, 50)
)

img_pil = Image.fromarray(img_cv)
draw = ImageDraw.Draw(img_pil)
detected_results = []

update_progress(40, "AI 사전 매칭 중")
text_features, target_labels = get_text_embeddings(is_demo_mode)

total_faces = len(faces)
for i, (x, y, w, h) in enumerate(faces):
current_prog = 40 + int(50 * ((i + 1) / max(1, total_faces)))
update_progress(current_prog, f"AI가 시각적 특징에서 단어를 추론 중 ({i+1}/{total_faces})")

x, y = max(0, x), max(0, y)
face_img = img_cv[y:y+h, x:x+w]
if face_img.size == 0: continue

face_pil = Image.fromarray(face_img)
inputs = processor(images=face_pil, return_tensors="pt")

with torch.no_grad():
image_outputs = model.get_image_features(**inputs)

if hasattr(image_outputs, "pooler_output"):
image_features = image_outputs.pooler_output
if image_features.shape[-1] != 512 and hasattr(model, "visual_projection"):
image_features = model.visual_projection(image_features)
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
person_results = []

for idx in top_indices:
label_data = target_labels[idx]
eng_word = label_data["word"]
kor_word = label_data["kor_word"]
kor_def = label_data["def"]
is_unsafe = label_data["is_unsafe"]

# chink 단어의 한국어 의미 강제 변경
if eng_word.lower() == "chink":
kor_word = "눈 찢어진 동양인"

if is_unsafe:
is_face_unsafe = True

if not is_unsafe and not kor_word:
kor_word = get_realtime_translation(eng_word)

display_box_text = f"{eng_word}({kor_word})" if kor_word else eng_word
display_texts.append(display_box_text)

if is_unsafe:
detail_text = f"🚨 {eng_word}({kor_word}) : {kor_def}" if kor_def else f"🚨 {eng_word}({kor_word})"
res_dict = {"text": detail_text, "type": "unsafe"}
else:
detail_text = f"✅ {eng_word}({kor_word})" if kor_word else eng_word
res_dict = {"text": detail_text, "type": "safe"}

person_results.append(res_dict)

detected_results.append({"person": i + 1, "labels": person_results})

box_color = (255, 0, 0) if is_face_unsafe else (0, 200, 80)
display_box_text_combined = f"[인물 {i+1}]\n" + "\n".join(display_texts)

draw.rectangle([(x, y), (x+w, y+h)], outline=box_color, width=dynamic_thickness)
bbox = draw.multiline_textbbox((x, y), display_box_text_combined, font=dynamic_font)
text_w = bbox[2] - bbox[0]
text_h = bbox[3] - bbox[1]
draw.rectangle(
[(x, y - text_h - int(dynamic_thickness*3)), (x + text_w + int(dynamic_thickness*2), y)],
fill=box_color
)
draw.multiline_text(
(x + 2, y - text_h - int(dynamic_thickness*2)),
display_box_text_combined,
font=dynamic_font,
fill=(0, 0, 0) if box_color == (0, 200, 80) else (255, 255, 255)
)

update_progress(100, "분석 완료")
time.sleep(0.4)
return img_pil, detected_results

# ---------------------------------------------------------
# 6. Stream디오 화면 UI
# ---------------------------------------------------------

st.markdown("<div class='eyebrow'>ImageNet 2011 학습 데이터 기반</div>", unsafe_allow_html=True)
st.markdown("<h1>AI 얼굴 인식 라벨링 테스트</h1>", unsafe_allow_html=True)
st.markdown("""
<div class='subtitle'>
본 테스트는 인간의 편견을 학습한 AI를 보여주는 시뮬레이션입니다.<br>
전 세계 얼굴 인식 AI의 훈련장으로 쓰이는 IMAGENET의 실제 과거 카테고리 분류를 활용해<br>인물 사진과 매칭되는 단어를 보여줍니다.
</div>
""", unsafe_allow_html=True)

if os.path.exists("img.jpg"):
st.image("img.jpg", use_container_width=True)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

demo_mode = st.session_state.get("demo_mode_toggle", False)

option = st.radio(
"이미지 입력 방식을 선택하세요:",
("웹캠 캡처", "사진 업로드"),
horizontal=True
)

image_to_process = None
upload_error = None

if option == "웹캠 캡처":
camera_image = st.camera_input("웹캠을 연결하고 사진을 찍어보세요.")
if camera_image is not None:
image_to_process, upload_error = load_and_prep_image(camera_image)

elif option == "사진 업로드":
uploaded_file = st.file_uploader(
"얼굴이 나온 사진을 업로드하세요.",
help="대부분의 이미지 형식(JPG, PNG, WebP, HEIC 호환 등)을 지원합니다."
)
if uploaded_file is not None:
image_to_process, upload_error = load_and_prep_image(uploaded_file)

if upload_error:
st.markdown(f"""
   <div class='upload-error'>
       ⚠️ <b>이미지를 불러올 수 없습니다</b><br>
       {upload_error}<br><br>
       <span style='font-size:0.82rem; color:#888;'>
       iPhone 사용자: 설정 → 카메라 → 포맷 → 가장 호환성 높은 포맷으로 변경 후 촬영하면 시스템 오류가 해결됩니다.
       </span>
   </div>
   """, unsafe_allow_html=True)

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
for person_data in results:
st.markdown(f"<div class='person-header'>👤 인물 {person_data['person']}</div>", unsafe_allow_html=True)
for res in person_data['labels']:
if res["type"] == "unsafe":
st.markdown(f"<div class='result-box unsafe-box'>{res['text']}</div>", unsafe_allow_html=True)
else:
st.markdown(f"<div class='result-box safe-box'>{res['text']}</div>", unsafe_allow_html=True)

st.markdown("""
       <div class='legend-box'>
           🚨 이미지넷이 공식적으로 판단한 노골적인 혐오/차별 단어 &nbsp;·&nbsp;
           ✅ 현재까지 이미지넷 분류 체계에 남아있는 인물 규정 단어
       </div>
       """, unsafe_allow_html=True)
else:
st.info("얼굴이 명확하게 인식되지 않았습니다. 조명이 밝은 곳에서 정면을 응시해 주세요.")

if not st.session_state.history or st.session_state.history[-1]["results"] != results:
st.session_state.history.append({
"image": processed_image,
"results": results
})

# ---------------------------------------------------------
# 7. 하단 UI: 과거 기록 → 단어 리스트 → 체험 스위치 → 논란 설명
# ---------------------------------------------------------

if st.session_state.history:
st.markdown("<br><hr class='divider'>", unsafe_allow_html=True)
st.subheader("과거 분석 기록")

cols = st.columns(4)
for idx, item in enumerate(reversed(st.session_state.history)):
col = cols[idx % 4]
with col:
st.image(item["image"], use_container_width=True)
if item["results"]:
for person_data in item["results"]:
st.markdown(f"<div class='history-text' style='color:#444; font-weight:600; margin-top:6px;'>[인물 {person_data['person']}]</div>", unsafe_allow_html=True)
for res in person_data['labels']:
short_text = res["text"].split(" : ")[0]
color = "#c0392b" if res["type"] == "unsafe" else "#2e7d45"
st.markdown(f"<div class='history-text' style='color:{color};'>{short_text}</div>", unsafe_allow_html=True)
else:
st.markdown("<div class='history-text' style='color:#aaa;'>미인식</div>", unsafe_allow_html=True)

unsafe_items = [item for item in BIAS_LABELS if item["is_unsafe"]]

# 단어 리스트 ABC 오름차순 정렬
unsafe_items = sorted(unsafe_items, key=lambda x: x["word"].lower())

# 리스트에서 'abbe', 'abbess' 제거 로직
filtered_unsafe_items = [item for item in unsafe_items if item["word"].lower() not in ["abbe", "abbess"]]

st.markdown("<br>", unsafe_allow_html=True)
# 개수는 하드코딩으로 총 1,593개 유지
st.subheader(f"🚨 문제적 편견/혐오 단어 리스트 (총 1,593개)")
st.caption("AI의 얼굴인식 학습 분류에 사용된 실제 혐오/편견 단어들입니다.")

unsafe_html = "<div class='word-list-container'><ul style='list-style-type:none; padding-left:0; margin:0;'>"
# 필터링된 리스트로 출력
for item in filtered_unsafe_items:
word = item["word"]
kor_word = item["kor_word"]

# chink 단어의 한국어 의미 강제 변경
if word.lower() == "chink":
kor_word = "눈 찢어진 동양인"

kor_def = item["def"]
unsafe_html += f"<li style='color:#c0392b; margin-bottom:6px; font-size:0.82em;'>⚠ <b>{word}({kor_word})</b> : {kor_def}</li>"
unsafe_html += "</ul></div>"
st.markdown(unsafe_html, unsafe_allow_html=True)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

st.markdown("<h4 style='text-align:center; margin-bottom:1rem;'>⚙ 체험 모드 설정</h4>", unsafe_allow_html=True)

# st.columns를 사용하여 토글을 완벽하게 가운데 정렬
col_dummy1, col_toggle, col_dummy2 = st.columns([1, 2, 1])
with col_toggle:
st.toggle("🚨 극단적 편향 모드 켜기(부정적/편견단어만\u00A0매칭)", key="demo_mode_toggle")

if st.session_state.get("demo_mode_toggle", False):
st.error("⚠️ 이 모드에서는 편향성을 학습한 AI를 보여주기 위해 대상의 특징을 혐오 단어로만 표시합니다.", icon="🚨")

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

st.subheader("LG 구겐하임 어워드 수상자 트레버 페글렌, AI의 보이지 않는 구조에 질문을 던지다")
st.markdown("""
<div class='article-text'>
<b>인공지능(AI)이 당신의 외모를 분석해 한 단어로 규정한다면 어떨까.</b>
<br><br>
피부색과 성별, 옷차림만으로 당신은 범죄자, 알코올 중독자라는 낙인이 붙을 수도 있다. 트레버 페글렌은 2019년 관객 참여형 프로젝트 '이미지넷 룰렛(ImageNet Roulette)'을 진행하며 AI가 인간을 분류하는 방식에 내재된 편견을 가감 없이 드러냈다.
<br><br>
이미지넷이란 세계 최대 규모의 이미지 데이터베이스로 1,400만개 이상의 이미지와 2만개가 넘는 카테고리 분류를 두고 있다. 지금도 전 세계 개발자들이 이미지넷을 인공지능의 훈련장으로 쓰며 이미지 학습과 얼굴 인식 AI의 기반으로 활용하고 단어로 규정한다면 어떨까.
<br><br>
<b>미디어 아티스트 트레버 페글렌(Trevor Paglen, 미국, 1974년생)</b>은 이미지넷의 판단 알고리즘을 그대로 가져와 사람들이 직접 자신의 셀카를 업로드하면, 인공지능이 어떻게 사람을 분류하는지를 실시간으로 보여줬다. 흑인 남성은 범죄자, 용의자로 분류되고, 안경 쓴 사람은 괴짜, 공부벌레 같은 라벨이 붙는 식이다. 그는 오염된 학습 데이터 속 AI가 내리는 판단에 인간이 가진 편견과 인종 차별이 녹아들어가 있음을 경고했다.
<br><br>
소셜 미디어를 통해 확산되며 논란이 거세지자, 결국 이미지넷을 관리하던 연구팀은 <b>2019년 9월, 공식 사과와 함께 1,593개의 혐오·차별적 표현을 데이터베이스에서 전면 삭제</b>했다. 해당 카테고리에서만 60만장의 사진이 삭제되었고 분류 체계가 전면 수정되며 AI 업계에 변화를 이끌어 냈다.
<br><br>
<b>AI 등 기술의 시선에 질문을 던져온 트레버 페글렌이 'LG 구겐하임 어워드' 수상자로 선정된 것은 우연이 아니다.</b> LG 역시 기술이 단순히 효율성을 높이는 도구를 넘어 인간의 삶에 어떤 영감과 영향을 주는지를 고민해 왔기 때문이다.
<br><br>
LG가 추구하는 '책임 있는 AI' 철학은 기술의 윤리적 의미를 성찰하는 페글렌의 작품 세계와 연결된다.
<br><br>
LG는 전 세계 기업 중 유일하게 유네스코 AI 윤리 권고 이행 현황을 공개하고 2023년부터 매년 'AI 윤리 책무성 보고서'도 발간하며 매년 기술의 투명성과 책임성을 대외적으로 공표하고 있다.
<br><br>
자체 파운데이션 AI 모델 엑사원(EXAONE) 개발에도 AI 위험분류 체계를 적용해 검증하고 있다. LG AI연구원이 개발한 범용 AI 위험분류체계 한국판(KAUT, Korea-Augmented Universal Taxonomy)은 잠재적 위험을 ▲인류 보편적 가치 ▲사회 안전 ▲한국적 특수성 ▲미래 위험 등 4개 핵심 영역, 226개 세부 위험 항목으로 구성되어 있으며, 항목별 5가지 구체적 판별 기준이 있어 하나의 위반 사항만 발생해도 AI가 부적절한 응답을 했다고 분류한다.
<br><br>
LG 관계자는 <b>"트레버 페글렌의 작품에 투영된 질문들은 LG가 해왔던 고민과 같은 선상에 있다"며 "LG 역시 AI 역량을 강화해 나감에 있어 투명성, 책임성, 그리고 기술의 인간 중심적 활용이 진정한 혁신의 기반이라고 믿는다"고 말했다. 이어 "그의 수상을 축하하며, 인간의 신뢰를 받을 수 있는 AI 미래를 구축하겠다는 LG의 의지를 다시 한번 다지는 계기가 될 것"</b>이라고 말했다.
</div>
""", unsafe_allow_html=True)
