import streamlit as st
import pandas as pd
import numpy as np
import cv2
from datetime import datetime

st.set_page_config(page_title="CSRIS - Video Crowd Counter", layout="wide")
st.title("🚨 CSRIS - Video-Based Crowd Risk Detection")

# Show current time
st.sidebar.markdown(f"**Current Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Upload video
st.header("📽 Upload Crowd Video")
video_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

# Crowd detection and counting logic
if video_file:
    with open("temp_video.mp4", "wb") as f:
        f.write(video_file.read())

    st.success("Video uploaded successfully. Processing...")

    cap = cv2.VideoCapture("temp_video.mp4")
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    total_people_count = 0
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 360))
        boxes, _ = hog.detectMultiScale(frame, winStride=(8, 8))
        total_people_count += len(boxes)

        for (x, y, w, h) in boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(rgb_frame, channels="RGB", use_column_width=True)

    cap.release()
    st.success(f"✅ Estimated Total People Detected: {total_people_count}")
else:
    st.info("Please upload a video to begin analysis.")

# Guidelines
st.header("📋 NDMA Guidelines")
st.markdown("""
- Safe: **≤ 4 people/m²**  
- Warning: **4–7 people/m²**  
- High risk: **> 7 people/m²**  

**Recommendations:**  
- Use AI surveillance to detect and respond in real-time  
- Equip venues with automated crowd alert systems  
- Notify responders and display alerts on large screens  
""")
