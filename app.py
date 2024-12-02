import speech_recognition as sr
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import av
import threading

# Load the emotion detection model
pipe_lr = joblib.load(open("text_emotion.pkl", "rb"))

emotions_emoji_dict = {"anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", "joy": "😂", "neutral": "😐", 
                       "sad": "😔", "sadness": "😔", "shame": "😳", "surprise": "😮"}

def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.text = None
        self.lock = threading.Lock()  # To manage threading safely

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        raw_audio = frame.to_ndarray().flatten()
        audio_data = np.int16(raw_audio).tobytes()

        def process_audio():
            audio_stream = sr.AudioData(audio_data, frame.sample_rate, 2)
            try:
                with self.lock:
                    self.text = self.recognizer.recognize_google(audio_stream)
            except sr.UnknownValueError:
                self.text = "Could not understand audio"
            except sr.RequestError:
                self.text = "API unavailable"

        # Run audio processing in a separate thread
        threading.Thread(target=process_audio).start()

        return frame

def process_text(raw_text):
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

def load_css():
    with open("style.css") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def main():
    load_css()
  
    st.title("SceneSonic")
    st.subheader("A cutting-edge AI platform designed to revolutionize how emotions are understood in theater!")

    option = st.selectbox("Choose Input Method", ("Record Voice", "Type Text"))

    if option == "Type Text":
        with st.form(key='my_form'):
            raw_text = st.text_area("Type Here")
            submit_text = st.form_submit_button(label='Submit')

        if submit_text:
            process_text(raw_text)

    elif option == "Record Voice":
        webrtc_ctx = webrtc_streamer(
            key="speech-to-text",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"audio": True, "video": False},
            audio_processor_factory=AudioProcessor,
        )

        if webrtc_ctx.state.playing and webrtc_ctx.audio_processor:
            if st.button("Stop"):
                raw_text = webrtc_ctx.audio_processor.text
                if raw_text and raw_text != "Could not understand audio":
                    st.write("Transcribed Text: ", raw_text)
                    process_text(raw_text)

if __name__ == '__main__':
    main()
