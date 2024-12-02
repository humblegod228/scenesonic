import speech_recognition as sr
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
import librosa
import pickle
from streamlit_webrtc import AudioProcessorBase, WebRtcMode
import av
import threading

# ================================
# Load Emotion Detection Model
# ================================

# Text emotion detection model
pipe_lr = joblib.load(open("text_emotion.pkl", "rb"))

# Emojis for emotions
emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", "joy": "😂",
    "neutral": "😐", "sad": "😔", "sadness": "😔", "shame": "😳", "surprise": "😮"
}

# Load audio emotion recognition model and scaler
def load_emotion_recognition_model(model_path, scaler_path):
    from keras.models import load_model  # Ensure this import exists
    model = load_model(model_path)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

# ================================
# Feature Extraction for Audio
# ================================

def extract_features(data, sample_rate):
    """
    Extract audio features like ZCR, Chroma, MFCC, RMS, and MelSpectrogram.
    """
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
    """
    Extract features from audio in chunks for better processing.
    """
    data, sample_rate = librosa.load(file_path)
    total_duration = librosa.get_duration(y=data, sr=sample_rate)

    if total_duration < chunk_size:
        return extract_features(data, sample_rate).reshape(1, -1)
    
    all_features = []
    for i in range(0, int(total_duration // chunk_size)):
        start = int(i * chunk_size * sample_rate)
        end = int((i + 1) * chunk_size * sample_rate)
        chunk = data[start:end]
        all_features.append(extract_features(chunk, sample_rate))
    
    return np.vstack(all_features) if all_features else np.array([]).reshape(0, 0)

# ================================
# Predictions for Text and Audio
# ================================

def predict_emotions(docx):
    """
    Predict emotion from text input.
    """
    return pipe_lr.predict([docx])[0]

def get_prediction_proba(docx):
    """
    Get prediction probabilities for text input.
    """
    return pipe_lr.predict_proba([docx])

def predict_emotion_from_audio(file_path, model, scaler):
    """
    Predict emotion from an audio file using a pre-trained model.
    """
    emotion_mapping = {
        0: 'angry', 1: 'calm', 2: 'disgust', 3: 'fear',
        4: 'happy', 5: 'neutral', 6: 'sad', 7: 'surprise'
    }

    features = get_features_from_audio(file_path)
    scaled_features = scaler.transform(features)
    reshaped_features = np.expand_dims(scaled_features, axis=2)

    predictions = model.predict(reshaped_features)  # Predictions for each chunk
    averaged_predictions = np.mean(predictions, axis=0)  # Average probabilities across chunks

    # Get the most probable emotion
    predicted_index = np.argmax(averaged_predictions)
    predicted_emotion = emotion_mapping[predicted_index]
    confidence = averaged_predictions[predicted_index]

    # Map predictions to emotion probabilities
    emotion_probabilities = {emotion_mapping[i]: prob for i, prob in enumerate(averaged_predictions)}

    return predicted_emotion, confidence, emotion_probabilities

# ================================
# Streamlit Frontend
# ================================

def load_css():
    """
    Load custom CSS for styling.
    """
    with open("style.css") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def process_text(raw_text):
    """
    Process text input and display predictions.
    """
    col1, col2 = st.columns(2)
    prediction = predict_emotions(raw_text)
    probability = get_prediction_proba(raw_text)

    with col1:
        st.success("Original Text")
        st.write(raw_text)

        st.success("Prediction")
        emoji_icon = emotions_emoji_dict[prediction]
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
    """
    Main Streamlit application logic.
    """
    load_css()
    st.title("SceneSonic")
    st.subheader("Revolutionize how emotions are understood in theater!")

    # Choose input method
    option = st.selectbox("Choose Input Method", ("Upload an audio file", "Type Text"))

    if option == "Type Text":
        with st.form(key='text_form'):
            raw_text = st.text_area("Type Here")
            submit_text = st.form_submit_button(label='Submit')

        if submit_text:
            process_text(raw_text)

    elif option == "Upload an audio file":
        uploaded_file = st.file_uploader("Choose an audio file...", type=["wav", "mp3"])
        if uploaded_file is not None:
            with st.spinner("Processing audio..."):
                with open("temp_audio.wav", "wb") as f:
                    f.write(uploaded_file.getbuffer())

            # Load models
                model, scaler = load_emotion_recognition_model('model/emotion_recognition_model(1).h5', 'model/scaler(1).pkl')

            # Predict emotion
                predicted_emotion, confidence, emotion_probabilities = predict_emotion_from_audio("temp_audio.wav", model, scaler)

            # Display results
                st.success(f"Predicted Emotion: **{predicted_emotion}**")
                st.info(f"Confidence: {confidence:.2f}")

            # Show probabilities as a bar chart
                st.subheader("Prediction Probabilities")
                proba_df = pd.DataFrame(list(emotion_probabilities.items()), columns=["Emotion", "Probability"])
                fig = alt.Chart(proba_df).mark_bar().encode(
                    x=alt.X("Emotion", sort=None),
                    y="Probability",
                    color="Emotion"
                )
                st.altair_chart(fig, use_container_width=True)

if __name__ == '__main__':
    main()
