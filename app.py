import numpy as np
import librosa
import pickle
from keras.models import load_model
import streamlit as st
from audio_recorder_streamlit import audio_recorder
import joblib
import pandas as pd
import altair as alt

# Load models and scaler
pipe_lr = joblib.load(open("text_emotion.pkl", "rb"))
def load_emotion_recognition_model(model_path, scaler_path):
    model = load_model(model_path)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

# Load the audio model and scaler
model, scaler = load_emotion_recognition_model('model/emotion_recognition_model(1).h5', 'model/scaler(1).pkl')

# Emotion mapping for audio classification
emotion_mapping = {
    0: 'angry',
    1: 'calm',
    2: 'disgust',
    3: 'fear',
    4: 'happy',
    5: 'neutral',
    6: 'sad',
    7: 'surprise'
}

emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", "joy": "😂", "neutral": "😐",
    "sad": "😔", "sadness": "😔", "shame": "😳", "surprise": "😮"
}

def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

def extract_features(data, sample_rate):
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc))
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))
    return result

def get_features_from_audio(file_path, chunk_size=2.5):
    data, sample_rate = librosa.load(file_path)
    total_duration = librosa.get_duration(y=data, sr=sample_rate)
    if total_duration < chunk_size:
        return extract_features(data, sample_rate).reshape(1, -1)
    all_features = []
    for i in range(0, int(total_duration // chunk_size)):
        start = int(i * chunk_size * sample_rate)
        end = int((i + 1) * chunk_size * sample_rate)
        chunk = data[start:end]
        res1 = extract_features(chunk, sample_rate)
        all_features.append(res1)
    if len(all_features) > 0:
        result = np.vstack(all_features)
    else:
        result = np.array([]).reshape(0, 0)
    return result

def predict_emotion_from_audio(file_path, model, scaler):
    features = get_features_from_audio(file_path)
    scaled_features = scaler.transform(features)
    reshaped_features = np.expand_dims(scaled_features, axis=2)
    predictions = model.predict(reshaped_features)
    predicted_indices = np.argmax(predictions, axis=1)
    predicted_emotions = [emotion_mapping[i] for i in predicted_indices]
    unique, counts = np.unique(predicted_emotions, return_counts=True)
    final_prediction = unique[np.argmax(counts)]
    return final_prediction

def process_text(raw_text):
    col1, col2 = st.columns(2)
    prediction = predict_emotions(raw_text)
    probability = get_prediction_proba(raw_text)
    with col1:
        st.success("Original Text")
        st.write(raw_text)
        st.success("Prediction")
        emoji_icon = emotions_emoji_dict.get(prediction, "")
        st.write(f"{prediction}: {emoji_icon}")
        st.write(f"Confidence: {np.max(probability)}")
    with col2:
        st.success("Prediction Probability")
        proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
        proba_df_clean = proba_df.T.reset_index()
        proba_df_clean.columns = ["emotions", "probability"]
        fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
        st.altair_chart(fig, use_container_width=True)

def main():
    st.title("SceneSonic")
    st.subheader("A cutting-edge AI platform designed to revolutionize how emotions are understood in theater!")
    option = st.selectbox("Choose Input Method", ("Type Text", "Upload Audio"))

    if option == "Type Text":
        with st.form(key='text_form'):
            raw_text = st.text_area("Type Here")
            submit_text = st.form_submit_button(label='Submit')
        if submit_text:
            process_text(raw_text)

    elif option == "Upload Audio":
        st.subheader("Upload an Audio File")
        uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])
        if uploaded_file is not None:
            with st.spinner("Processing audio..."):
                with open("temp_audio.wav", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                predicted_emotion = predict_emotion_from_audio("temp_audio.wav", model, scaler)
                st.success(f"Predicted Emotion: **{predicted_emotion}**")

if __name__ == "__main__":
    main()
