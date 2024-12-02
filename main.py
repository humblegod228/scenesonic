import numpy as np
import librosa
import pickle
from keras.models import load_model
import streamlit as st
from audio_recorder_streamlit import audio_recorder

# Load the trained model, scale
def load_emotion_recognition_model(model_path, scaler_path):
    model = load_model(model_path)
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
        
    
    return model, scaler

def extract_features(data, sample_rate):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc))

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))
    
    return result

def get_features_from_audio(file_path, chunk_size=2.5):
    data, sample_rate = librosa.load(file_path)
    total_duration = librosa.get_duration(y=data, sr=sample_rate)

    if total_duration < chunk_size:
        # If the audio is too short, use the entire audio for feature extraction
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
        # If no features were extracted, return an empty array
        result = np.array([]).reshape(0, 0)
        
    return result

def predict_emotion_from_audio(file_path, model, scaler):
    # Use the categories you provided
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
    
    features = get_features_from_audio(file_path)
    scaled_features = scaler.transform(features)
    reshaped_features = np.expand_dims(scaled_features, axis=2)
    
    predictions = model.predict(reshaped_features)
    
    # Convert the predictions to emotion labels
    predicted_indices = np.argmax(predictions, axis=1)
    predicted_emotions = [emotion_mapping[i] for i in predicted_indices]
    
    # Count the most frequent emotion in the predictions
    unique, counts = np.unique(predicted_emotions, return_counts=True)
    final_prediction = unique[np.argmax(counts)]
    
    return final_prediction

def main():
    # Load the model, scaler
    model, scaler = load_emotion_recognition_model('model/emotion_recognition_model(1).h5', 'model/scaler(1).pkl')

    # Streamlit UI
    st.title("🎙️ Emotion Recognition from Audio")

    option = st.selectbox("Choose an option", ["Upload an audio file", "Record audio"])

    if option == "Upload an audio file":
        uploaded_file = st.file_uploader("Choose an audio file...", type=["wav", "mp3"])

        if uploaded_file is not None:
            with st.spinner("Processing audio..."):
                # Save the uploaded file to a temporary location
                with open("temp_audio.wav", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Predict emotion
                predicted_emotion = predict_emotion_from_audio("temp_audio.wav", model, scaler)
                
                st.success(f"Predicted Emotion: **{predicted_emotion}**")

    elif option == "Record audio":
        
        audio_bytes = audio_recorder()
        
        if audio_bytes:
            st.audio(audio_bytes, format='audio/wav')
            with st.spinner("Processing recorded audio..."):
                # Save the recorded audio to a file
                with open("temp_audio.wav", "wb") as f:
                    f.write(audio_bytes)
                
                # Predict emotion
                predicted_emotion = predict_emotion_from_audio("temp_audio.wav", model, scaler)
                
                st.success(f"Predicted Emotion: **{predicted_emotion}**")

if __name__ == "__main__":
    main()
