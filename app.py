# app.py
import streamlit as st
import cv2
import numpy as np
import utils
import time
import threading
from utils import DrowsinessDetector
import mediapipe as mp
from PIL import Image
import time

st.set_page_config(page_title="Driver Drowsiness Detection", layout="wide")

st.title("Driver Drowsiness & Distraction Detection")
col1, col2 = st.columns([2,1])

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    ear_thresh = st.slider("EAR threshold", 0.12, 0.35, 0.22, 0.01)
    ear_frames = st.slider("EAR consecutive frames", 1, 60, 15, 1)
    mar_thresh = st.slider("MAR threshold", 0.2, 1.0, 0.6, 0.05)
    mar_frames = st.slider("MAR consecutive frames", 1, 10, 3, 1)
    head_angle = st.slider("Head angle threshold (deg)", 5, 60, 20, 1)
    start_button = st.button("Start Webcam")
    stop_button = st.button("Stop")

# placeholders
video_ph = col1.empty()
status_ph = col2.empty()
log_ph = col2.empty()

detector = DrowsinessDetector(
    ear_thresh=ear_thresh,
    ear_consec_frames=ear_frames,
    mar_thresh=mar_thresh,
    mar_consec_frames=mar_frames,
    head_angle_thresh=head_angle
)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

# audio alarm helper
def play_alarm():
    # simple cross-platform beep as fallback
    try:
        import simpleaudio as sa
        freq = 440
        fs = 44100
        t = 0.5
        samples = (np.sin(2*np.pi*np.arange(int(fs*t))*freq/fs)).astype(np.float32)
        sa.play_buffer((samples*32767).astype(np.int16), 1, 2, fs)
    except Exception:
        try:
            import winsound
            winsound.Beep(1000, 500)
        except Exception:
            pass

st.info("Press Start Webcam to run the detection. Use Stop to end.")

run = False
cap = None

if start_button:
    run = True
    cap = cv2.VideoCapture(0)
    st.session_state["run"] = True

if "run" in st.session_state and st.session_state["run"]:
    run = True

if stop_button:
    run = False
    st.session_state["run"] = False
    if cap:
        cap.release()
        cap = None

# main loop
if run:
    if cap is None:
        cap = cv2.VideoCapture(0)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Couldn't read from webcam")
            break
        frame = cv2.flip(frame, 1)  # mirror
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        h, w, _ = frame.shape

        ear = 1.0
        mar = 0.0
        head_angle = 0.0

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            lm = []
            for i, p in enumerate(landmarks.landmark):
                lm.append([p.x * w, p.y * h])
            lm = np.array(lm)

            # select eye landmarks (MediaPipe face mesh indexes)
            left_eye_idx = [33, 160, 158, 133, 153, 144]
            right_eye_idx = [362, 385, 387, 263, 373, 380]
            left_eye = lm[left_eye_idx]
            right_eye = lm[right_eye_idx]
            ear_left = utils.eye_aspect_ratio(left_eye)
            ear_right = utils.eye_aspect_ratio(right_eye)
            ear = (ear_left + ear_right) / 2.0

            # mouth indices (MediaPipe): lower/upper inner
            mouth_idx = [78, 95, 88, 178, 87, 14, 317, 402]
            mouth = lm[mouth_idx]
            mar = utils.mouth_aspect_ratio(mouth)

            head_angle = utils.head_pose_score(lm)

            # draw landmarks for debug
            mp_drawing.draw_landmarks(frame, landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                                      mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
                                      mp_drawing.DrawingSpec(color=(0,0,255), thickness=1))

        # update detector
        state = detector.update(ear=ear, mar=mar, head_angle=head_angle)

        # display status and overlay text
        status_text = f"Status: {state['status']} | EAR: {ear:.3f} | MAR: {mar:.3f} | HeadAngle: {head_angle:.1f}"
        color = (0,255,0)
        if state['status'] == "Drowsy":
            color = (0,0,255)
            # play alarm asynchronously to avoid blocking UI
            threading.Thread(target=play_alarm, daemon=True).start()
        elif state['status'] == "Distracted":
            color = (0,165,255)
        cv2.putText(frame, status_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # convert to display
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_ph.image(img_rgb, channels="RGB")

        # sidebar display
        with col2:
            status_ph.markdown(f"### Current status: **{state['status']}**")
            log_df = detector.event_log[::-1][:10]  # last 10 events
            if log_df:
                log_md = "\n".join([f"- {time.strftime('%H:%M:%S', time.localtime(e['time']))} : {', '.join(e['events'])} (EAR={e['ear']:.2f}, MAR={e['mar']:.2f})" for e in log_df])
                log_ph.markdown("**Recent events:**\n" + log_md)
            else:
                log_ph.markdown("**Recent events:**\n- None")

        # slow down loop slightly for CPU
        time.sleep(0.03)
        frame_count += 1

        # break condition from Streamlit (use Stop button)
        if "run" in st.session_state and not st.session_state["run"]:
            break
else:
    st.write("Webcam is stopped. Configure thresholds in the sidebar and click Start Webcam.")
