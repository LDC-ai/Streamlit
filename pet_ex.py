import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

# 딥러닝 모델 불러오기
model = tf.keras.models.load_model("best_model3.keras")

# 사진을 저장할 폴더 생성
if not os.path.exists("uploads"):
    os.makedirs("uploads")

# 과거에 테스트한 사진과 결과 저장
if 'past_results' not in st.session_state:
    st.session_state['past_results'] = []

# 사용자 선택 결과 저장
if 'user_selections' not in st.session_state:
    st.session_state['user_selections'] = []

# 이미지 전처리 함수
def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# 예측 함수
def predict_emotion(image):
    predictions = model.predict(image)[0]
    emotions = ['Angry', 'Happy', 'Sad', 'Other']
    emotion_confidences = {emotions[i]: predictions[i] * 100 for i in range(len(emotions))}
    predicted_emotion = max(emotion_confidences, key=emotion_confidences.get)
    return emotion_confidences, predicted_emotion

# HTML을 이용해 배경색 설정
page_bg = """
<style>
    body {
        background-color: #f9c3ac;
    }
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# 웹 페이지 제목
st.title("🐾 Pet Emotion Classifier 🐾")

# 파일 업로드 인터페이스
uploaded_file = st.file_uploader("애완동물 사진을 업로드하세요", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 업로드된 사진을 저장하고 화면에 표시
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # 이미지 전처리 및 예측
    preprocessed_image = preprocess_image(image)
    emotion_confidences, predicted_emotion = predict_emotion(preprocessed_image)
    
    # 예측 결과 출력
    st.write("예측 결과:")
    for emotion, confidence in emotion_confidences.items():
        st.write(f"**{emotion}**: {confidence:.2f}%")
    
    st.write(f"가장 확률이 높은 감정: **{predicted_emotion}**")

    # 결과 저장 (이미지 업로드 시에만 저장)
    image_path = os.path.join("uploads", uploaded_file.name)
    image.save(image_path)
    st.session_state['past_results'].insert(0, (image_path, predicted_emotion, emotion_confidences[predicted_emotion]))  # 최신 결과를 위에 추가

    # 예측 확률 차트 선택
    chart_type = st.selectbox("확률 차트 유형 선택:", ["Bar Chart", "Pie Chart"])
    
    if chart_type == "Bar Chart":
        plt.figure(figsize=(8, 4))
        plt.bar(emotion_confidences.keys(), emotion_confidences.values(), color='blue')
        plt.xlabel('Emotions')
        plt.ylabel('Confidence (%)')
        plt.title('Emotion Prediction Confidence (Bar Chart)')
        st.pyplot(plt)

    elif chart_type == "Pie Chart":
        plt.figure(figsize=(6, 6))
        plt.pie(emotion_confidences.values(), labels=emotion_confidences.keys(), autopct='%1.1f%%', startangle=140)
        plt.title('Emotion Prediction Confidence (Pie Chart)')
        st.pyplot(plt)

# 오른쪽에 과거 테스트 결과 보여주기
st.sidebar.title("테스트한 이미지들")
st.sidebar.write("여태까지 테스트한 사진들입니다.")
for img_path, emotion, confidence in st.session_state['past_results']:
    st.sidebar.image(img_path, caption=f"{confidence:.2f}%로 {emotion}", use_column_width=True)