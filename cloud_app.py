import streamlit as st
import numpy as np
import time

st.set_page_config(page_title="Driver Drowsiness Detection (Demo)", layout="wide")

st.title("🚗 Driver Drowsiness & Distraction Detection (Demo)")
st.markdown("This is a simplified cloud demo. The full MediaPipe-based version runs locally.")

ear_thresh = st.slider("EAR threshold", 0.1, 0.35, 0.22, 0.01)
mar_thresh = st.slider("MAR threshold", 0.2, 1.0, 0.6, 0.05)

placeholder = st.empty()

statuses = ["Alert", "Drowsy", "Distracted"]
colors = {"Alert": "green", "Drowsy": "red", "Distracted": "orange"}

st.write("Simulating driver states…")

for i in range(100):
    status = np.random.choice(statuses, p=[0.6, 0.25, 0.15])
    with placeholder.container():
        st.markdown(f"<h2 style='color:{colors[status]};'>Status: {status}</h2>", unsafe_allow_html=True)
    time.sleep(1.0)
