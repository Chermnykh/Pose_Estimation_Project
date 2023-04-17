import streamlit as st
import base64

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.title("SmartFit")

st.subheader("Exercise Plan: \n * Squats \n * Curl")

audio_base64 = base64.b64encode(open("knees.wav", "rb").read()).decode('utf-8')
audio_tag = f'<audio autoplay="true" src="data:audio/wav;base64,{audio_base64}">'
st.markdown(audio_tag, unsafe_allow_html=True)
    



