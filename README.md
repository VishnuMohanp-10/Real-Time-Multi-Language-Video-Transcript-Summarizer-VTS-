# 🎥 Real-Time Multilingual Video Transcript Summarizer

An AI-powered full-stack web application that automatically transcribes and summarizes multilingual video content to improve accessibility for deaf and hard-of-hearing users.

---

## 🚀 Features

- 🎙 Multilingual Speech-to-Text Transcription
- 🧠 NLP-based Text Summarization
- 📂 Video Upload & Audio Extraction
- 🌐 REST API Backend (FastAPI)
- ⚡ Real-Time Processing (WebSockets - In Progress)

---

## 🏗 Architecture

Video Upload  
⬇  
Audio Extraction  
⬇  
Speech-to-Text Model (Whisper)  
⬇  
NLP Summarization  
⬇  
Structured Transcript + Summary Output  

---

## 🛠 Tech Stack

**Backend**
- Python
- FastAPI
- WebSockets
- SQLAlchemy
- SQLite

**Frontend**
- ReactJS
- HTML5, CSS3

**AI Models**
- Whisper (Speech Recognition)
- Transformer-based NLP Model

---

## 🎯 Objective

Designed to enhance digital accessibility by converting spoken video content into structured multilingual transcripts and concise summaries.
Enhance the real-time accessibility for deaf and hard-of-hearing users during live streaming sessions by providing real-time transcription and summarization in muliple languages.

---

## 📦 Setup Instructions

```bash
git clone https://github.com/VishnuMohanp-10/Real-Time-Multi-Language-Video-Transcript-Summarizer-VTS.git
cd Real-Time Multi-Language-VTS
pip install -r requirements.txt
uvicorn main:app --reload
