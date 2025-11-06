# Driver Drowsiness & Distraction Detection
Student: <Full Name>
Candidate Registration Number: <XXXXXX>
CRS Name: Artificial Intelligence
Course: Machine Learning and Deep Learning

## Project Overview
This project implements a real-time driver monitoring system that detects drowsiness and distraction using facial landmarks from MediaPipe and simple heuristics: Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), and head-pose score. The system raises visual and audible alerts and logs unsafe events.

## How to run (local)
1. Clone repo:
   ```bash
   git clone https://github.com/YourUsername/YourRepo.git
   cd YourRepo
   ```
2. Create and activate virtualenv:
   ```bash
   python -m venv venv
   source venv/bin/activate   # or .\venv\Scripts\activate on Windows
   ```
3. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```

## Notes
- On Streamlit Cloud you may not have webcam access; use the app's code to add video upload/demo mode if needed.
- Calibrate EAR/MAR thresholds for your dataset.
- Add screenshots and results to this README after testing.

## Files
- app.py – Streamlit app.
- utils.py – EAR/MAR/head_pose and stateful detector class.
- requirements.txt – Python dependencies.
- tests/ – unit tests for utility functions.
