# 🎬 Real-Time Multi-Language Video Transcript Summarizer (VTS)

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?style=for-the-badge&logo=fastapi)
![React](https://img.shields.io/badge/React-18+-61DAFB?style=for-the-badge&logo=react)
![Whisper](https://img.shields.io/badge/Faster_Whisper-ASR-412991?style=for-the-badge&logo=openai)
![WebSockets](https://img.shields.io/badge/WebSockets-Real--Time-orange?style=for-the-badge)
![Deployed](https://img.shields.io/badge/Deployed-Render-46E3B7?style=for-the-badge&logo=render)

**An AI-powered full-stack web application that transcribes, translates, and summarizes multilingual video content in real time — built to enhance accessibility for deaf and hard-of-hearing users.**

[🚀 Live Demo](https://real-time-multi-language-video-mnzl.onrender.com/) &nbsp;·&nbsp; [📧 Contact](mailto:vishnumohanp25@gmail.com) &nbsp;·&nbsp; [💼 LinkedIn](https://www.linkedin.com/in/vishnu-mohan-p/)

</div>

---

## 📹 Demo

https://github.com/user-attachments/assets/661ab328-8fda-4d5f-a041-db6c529797ce

---

## 🚀 Live App

👉 **[Try it here → real-time-multi-language-video-mnzl.onrender.com](https://real-time-multi-language-video-mnzl.onrender.com/)**

> ⚠️ Hosted on Render free tier — may take 30–60 seconds to wake up on first load.

---

## ✨ Features

- 🎙️ **Multilingual Speech-to-Text** — Real-time transcription with automatic language detection using Faster Whisper
- 🌍 **Multi-Language Translation** — Translates transcripts into multiple languages using Deep Translator
- 🧠 **NLP Summarization** — Generates concise, meaningful summaries from continuous speech input
- 📂 **Video Upload & Processing** — Upload video files with live audio extraction via FFmpeg
- 📡 **Live Stream Support** — Fetches and processes live streams via yt-dlp in real time
- ⚡ **WebSocket Streaming** — Instant transcript and summary updates pushed to the frontend as they are generated
- 🔐 **User Authentication** — Secure login with session management backed by SQLite
- 📥 **Download Transcripts** — Export transcripts and summaries for offline use

---

## 🏗️ System Architecture

```
Video Upload / Live Stream URL
        ⬇
Audio Extraction (FFmpeg / yt-dlp)
        ⬇
Speech-to-Text (Faster Whisper ASR — auto language detection)
        ⬇
Multi-Language Translation (Deep Translator)
        ⬇
NLP Summarization
        ⬇
Real-Time Output via WebSockets → React Frontend
        ⬇
Structured Transcript + Summary + Download
```

---

## 🛠️ Tech Stack

### 🤖 AI & Speech Processing
| Technology | Role |
|------------|------|
| **Faster Whisper** (ASR Model) | Real-time speech-to-text with automatic language detection |
| **NLP Summarization Model** | Generates concise summaries from continuous transcribed text |
| **FFmpeg** | Continuously extracts audio from video streams for processing |
| **yt-dlp** | Fetches live video streams and extracts streaming URLs |

### 🌐 Language Processing
| Technology | Role |
|------------|------|
| **Deep Translator** | Translates transcripts into multiple languages for accessibility |

### ⚙️ Backend & Communication
| Technology | Role |
|------------|------|
| **FastAPI** | Manages API requests and video processing workflows |
| **WebSockets** | Real-time bidirectional streaming of transcripts and summaries |
| **SQLAlchemy + SQLite** | Stores transcripts, summaries, user data, and session info |

### 🖥️ Frontend
| Technology | Role |
|------------|------|
| **React.js** | Interactive UI for video input and real-time result display |
| **HTML5, CSS3** | Responsive layout and styling |

---

## 🎯 Objective

Designed to enhance digital accessibility by converting spoken video content into structured multilingual transcripts and concise summaries.

Particularly focused on improving real-time accessibility for **deaf and hard-of-hearing users** during live streaming sessions — providing instant transcription, translation, and summarization in multiple languages as the video plays.

---

## ⚙️ Run Locally

```bash
# Clone the repository
git clone https://github.com/VishnuMohanp-10/Real-Time-Multi-Language-Video-Transcript-Summarizer-VTS-.git
cd Real-Time-Multi-Language-Video-Transcript-Summarizer-VTS-

# Install dependencies
pip install -r requirements.txt

# Start the backend
uvicorn main:app --reload

# Start the frontend (new terminal)
cd frontend
npm install
npm start
```

App runs at `http://localhost:3000`

---

## 👤 Contact

**Vishnu** — Full Stack & AI Developer

- 📧 [vishnumohanp25@gmail.com](mailto:vishnumohanp25@gmail.com)
- 💼 [linkedin.com/in/vishnu-mohan-p](https://www.linkedin.com/in/vishnu-mohan-p/)
- 🐙 [github.com/VishnuMohanp-10](https://github.com/VishnuMohanp-10)

---

<div align="center">
⭐ If you found this project useful, give it a star — it helps!
</div>
