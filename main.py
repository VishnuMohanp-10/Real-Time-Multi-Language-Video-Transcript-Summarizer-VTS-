import os
import tempfile
import time
import sqlite3
import hashlib
import secrets
import logging
import asyncio
import subprocess
import re
import signal
import random
import wave

import numpy as np
import soundfile as sf
import psutil

from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from deep_translator import GoogleTranslator
from faster_whisper import WhisperModel
import yt_dlp as youtube_dl


# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

shutdown_event = asyncio.Event()


# ============================================================================
# DATABASE — SQLite (no external DB needed, works on Render free tier)
# ============================================================================

DB_PATH = "vts_users.db"


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create all tables on first startup."""
    conn = get_db()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                username    TEXT    UNIQUE NOT NULL,
                password    TEXT    NOT NULL,
                created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
                last_login  TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                token       TEXT    PRIMARY KEY,
                user_id     INTEGER NOT NULL,
                username    TEXT    NOT NULL,
                created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS download_history (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id       INTEGER NOT NULL,
                username      TEXT    NOT NULL,
                language      TEXT    NOT NULL,
                content_type  TEXT    NOT NULL,
                downloaded_at TEXT    NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        conn.commit()
        logger.info("Database initialised")
    except Exception as e:
        logger.error(f"DB init error: {e}")
    finally:
        conn.close()


def hash_password(password: str) -> str:
    salt = "vts_secure_salt_2025"
    return hashlib.sha256(f"{salt}{password}".encode()).hexdigest()


def create_session_token() -> str:
    return secrets.token_hex(32)


def verify_token(token: str) -> Optional[dict]:
    if not token:
        return None
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT * FROM sessions WHERE token = ?", (token,)
        ).fetchone()
        return {"user_id": row["user_id"], "username": row["username"]} if row else None
    finally:
        conn.close()


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class LoginRequest(BaseModel):
    username: str
    password: str

class RegisterRequest(BaseModel):
    username: str
    password: str

class DownloadRequest(BaseModel):
    content: str        # full text to download
    language: str       # e.g. "english" / "en"
    content_type: str   # "transcript" | "translation" | "summary"
    token: str          # session token

class VideoProcessRequest(BaseModel):
    translate: bool = True
    summarize: bool = True

class TranscriptionItem(BaseModel):
    text: str
    timestamp: str

class StopStreamRequest(BaseModel):
    transcriptions: List[TranscriptionItem]
    language: Optional[str] = "en"

class GenerateSummaryRequest(BaseModel):
    transcriptions: List[TranscriptionItem]
    language: Optional[str] = "en"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def split_sentences_indic(text: str):
    """Sentence splitter that handles Indic danda (।) as well as . ? !"""
    sentences = re.split(r'(?<=[\.!\?।])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def is_repetition(text: str) -> bool:
    """Return True when text looks like hallucinated/repeated garbage."""
    tokens = text.split()
    if len(tokens) >= 3:
        if len(set(tokens)) / len(tokens) < 0.5:
            return True
    if len(text) > 10 and len(set(text)) < 5:
        return True
    return False


def summarize_text(text: str, max_sentences: int = 3, max_chars: int = 500) -> str:
    try:
        sentences = split_sentences_indic(text)
        if len(sentences) > max_sentences:
            summary = ' '.join(sentences[:max_sentences]) + '...'
            return summary[:max_chars] if len(summary) > max_chars else summary
        return text[:max_chars] if len(text) > max_chars else text
    except Exception as e:
        logger.error(f"Summarization error: {e}")
        return (text[:200] + '...') if len(text) > 200 else text


def get_video_duration(video_path: str) -> float:
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if result.returncode == 0 and result.stdout:
            return float(result.stdout.strip())
        raise ValueError("Could not determine video duration")
    except Exception as e:
        logger.error(f"Error getting video duration: {e}")
        raise ValueError("Could not determine video duration")


def extract_hls_url(info) -> Optional[str]:
    """Pick the best audio URL from a yt-dlp info dict."""
    if not info or not isinstance(info, dict):
        return None
    formats = info.get('formats', [])
    hls = [f for f in formats if f.get('protocol') == 'm3u8' or '.m3u8' in f.get('url', '')]
    if hls:
        audio_only = [f for f in hls if f.get('acodec') != 'none' and f.get('vcodec') == 'none']
        pool = audio_only if audio_only else hls
        return sorted(pool, key=lambda f: f.get('tbr', 0), reverse=True)[0]['url']
    working = [f for f in formats if f.get('url')]
    if working:
        return sorted(working, key=lambda f: f.get('tbr', 0), reverse=True)[0]['url']
    return None


async def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    try:
        translator = GoogleTranslator(source=source_lang, target=target_lang)
        result = await asyncio.to_thread(translator.translate, text)
        return result
    except Exception as e:
        logger.error(f"Translation {source_lang}->{target_lang} failed: {e}")
        return "Translation unavailable"


# ============================================================================
# YOUTUBE LIVE PROCESSOR
# ============================================================================

class YouTubeLiveProcessor:

    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    ]

    def __init__(self):
        self.stop_flag = False
        self.cookies_file = "cookies.txt"
        self.sample_rate = 16000
        self.detected_language = None
        self.max_retries = 5
        self.transcript_buffer: List[str] = []
        self.summary_buffer: List[str] = []
        self.final_summaries = {'en': '', 'hi': '', 'ml': '', 'ta': '', 'te': ''}
        self.ffmpeg_process = None
        self.ydl_process = None
        self.active_websocket = None
        self.current_url = None
        self.last_audio_time = datetime.now()
        self.connection_active = False
        self.lock = asyncio.Lock()
        self.audio_stall_timeout = 90

    # ── process management ────────────────────────────────────────────────

    def kill_processes(self):
        for attr in ('ffmpeg_process', 'ydl_process'):
            proc = getattr(self, attr, None)
            if proc:
                try:
                    proc.kill()
                except Exception:
                    pass
                setattr(self, attr, None)
        try:
            for proc in psutil.process_iter(['name']):
                if proc.info['name'] in ['ffmpeg', 'yt-dlp', 'yt-dlp.exe', 'ffmpeg.exe']:
                    try:
                        proc.kill()
                    except Exception:
                        pass
        except Exception as e:
            logger.error(f"kill_processes error: {e}")

    async def pipeline_monitor(self):
        await asyncio.sleep(15)
        while self.connection_active and not self.stop_flag and not shutdown_event.is_set():
            async with self.lock:
                dead = (
                    (self.ffmpeg_process and self.ffmpeg_process.poll() is not None) or
                    (self.ydl_process    and self.ydl_process.poll()    is not None)
                )
                stalled = (datetime.now() - self.last_audio_time).total_seconds() > self.audio_stall_timeout
                if dead or stalled:
                    reason = "process died" if dead else "audio stall (90s)"
                    logger.error(f"Pipeline monitor: {reason}")
                    self.kill_processes()
                    break
            await asyncio.sleep(15)

    async def read_stderr(self, pipe, prefix: str):
        buf = []
        loop = asyncio.get_event_loop()
        while self.connection_active and not self.stop_flag:
            try:
                line = await loop.run_in_executor(None, pipe.readline)
                if line:
                    buf.append(line.decode(errors='ignore').strip())
                    logger.debug(f"{prefix}: {buf[-1]}")
                else:
                    if buf:
                        logger.error(f"{prefix} output:\n" + "\n".join(buf[-10:]))
                    break
            except Exception as e:
                logger.error(f"read_stderr error: {e}")
                break

    # ── stream URL extraction — 6-strategy yt-dlp approach ───────────────

    async def get_stream_url_direct(self, video_id: str, websocket: WebSocket) -> Optional[str]:
        """
        Extract a playable audio stream URL using yt-dlp.
        Tries 6 YouTube client strategies in order until one succeeds.
        Always uses the canonical watch URL (strips ?si= and /live/ variants).
        """
        watch_url = f"https://www.youtube.com/watch?v={video_id}"
        logger.info(f"Extracting stream URL for: {watch_url}")

        # YouTube live HLS audio-only format IDs
        LIVE_FORMAT = "91/92/93/94/95/96/bestaudio/best"

        cookies_path = self.cookies_file if os.path.exists(self.cookies_file) else None

        strategies = [
            {
                "label": "android",
                "opts": {
                    "quiet": True, "no_warnings": True, "socket_timeout": 30,
                    "format": LIVE_FORMAT,
                    "extractor_args": {"youtube": {"player_client": ["android"]}},
                    "http_headers": {"User-Agent": "com.google.android.youtube/17.36.4 (Linux; U; Android 12) gzip"},
                }
            },
            {
                "label": "android_vr",
                "opts": {
                    "quiet": True, "no_warnings": True, "socket_timeout": 30,
                    "format": LIVE_FORMAT,
                    "extractor_args": {"youtube": {"player_client": ["android_vr"]}},
                    "http_headers": {"User-Agent": "com.google.android.apps.youtube.vr.oculus/1.56.21 (Linux; U; Android 12L; eureka-user Build/SQ3A.220605.009.A1) gzip"},
                }
            },
            {
                "label": "ios",
                "opts": {
                    "quiet": True, "no_warnings": True, "socket_timeout": 30,
                    "format": LIVE_FORMAT,
                    "extractor_args": {"youtube": {"player_client": ["ios"]}},
                    "http_headers": {"User-Agent": "com.google.ios.youtube/17.33.2 (iPhone14,3; U; CPU iOS 15_6 like Mac OS X)"},
                }
            },
            {
                "label": "tv_embedded",
                "opts": {
                    "quiet": True, "no_warnings": True, "socket_timeout": 30,
                    "format": LIVE_FORMAT,
                    "extractor_args": {"youtube": {"player_client": ["tv_embedded"], "player_skip": ["webpage"]}},
                }
            },
            {
                "label": "mweb",
                "opts": {
                    "quiet": True, "no_warnings": True, "socket_timeout": 45,
                    "format": LIVE_FORMAT,
                    "extractor_args": {"youtube": {"player_client": ["mweb"]}},
                    "http_headers": {"User-Agent": "Mozilla/5.0 (Linux; Android 12; Pixel 6) AppleWebKit/537.36 Mobile Safari/537.36"},
                }
            },
            {
                "label": "web",
                "opts": {
                    "quiet": True, "no_warnings": True, "socket_timeout": 60,
                    "format": LIVE_FORMAT,
                    "extractor_args": {"youtube": {"player_client": ["web"]}},
                    "http_headers": {
                        "User-Agent": random.choice(self.USER_AGENTS),
                        "Accept-Language": "en-US,en;q=0.9",
                    },
                }
            },
        ]

        for strategy in strategies:
            label = strategy["label"]
            opts = strategy["opts"].copy()
            if cookies_path:
                opts["cookiefile"] = cookies_path
            try:
                logger.info(f"  Trying {label}...")
                ydl = youtube_dl.YoutubeDL(opts)
                info = await asyncio.to_thread(ydl.extract_info, watch_url, download=False)
                if not info:
                    logger.warning(f"  {label}: no info returned")
                    continue
                url = info.get("url")
                if url:
                    logger.info(f"  {label}: got direct URL")
                    return url
                hls = extract_hls_url(info)
                if hls:
                    logger.info(f"  {label}: got HLS URL from formats")
                    return hls
                logger.warning(f"  {label}: info returned but no usable URL")
            except Exception as e:
                logger.warning(f"  {label} failed: {str(e)[:200]}")
                continue

        logger.error("All stream extraction strategies exhausted")
        return None

    # ── WAV writer ────────────────────────────────────────────────────────

    async def _write_buffer_to_wav(self, buffer_bytes: bytes) -> str:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.close()
        try:
            with wave.open(tmp.name, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)         # int16 = 2 bytes
                wf.setframerate(self.sample_rate)
                wf.writeframes(buffer_bytes)
            return tmp.name
        except Exception as e:
            logger.error(f"Failed to write WAV: {e}")
            try:
                os.unlink(tmp.name)
            except Exception:
                pass
            raise

    # ── main stream processing loop ───────────────────────────────────────

    async def process_stream(self, url: str, websocket: WebSocket, forced_language: Optional[str] = None):
        self.current_url = url
        logger.info(f"Starting stream processing: {url}")

        try:
            # ── 1. Extract clean 11-char video ID ────────────────────────
            video_id = None
            id_patterns = [
                r'youtube\.com/live/([a-zA-Z0-9_-]{11})',
                r'youtu\.be/([a-zA-Z0-9_-]{11})',
                r'[?&]v=([a-zA-Z0-9_-]{11})',
                r'youtube\.com/watch/([a-zA-Z0-9_-]{11})',
                r'youtube\.com/embed/([a-zA-Z0-9_-]{11})',
            ]
            for pattern in id_patterns:
                m = re.search(pattern, url)
                if m:
                    video_id = m.group(1)
                    break

            if not video_id:
                await websocket.send_json({
                    "type": "error",
                    "data": "Could not extract a valid YouTube video ID from the URL. Please check the link.",
                    "timestamp": datetime.now().isoformat()
                })
                return

            logger.info(f"Video ID: {video_id}")

            # ── 2. Resolve stream URL ─────────────────────────────────────
            await websocket.send_json({
                "type": "status",
                "data": "Resolving stream URL, please wait...",
                "timestamp": datetime.now().isoformat()
            })

            hls_url = await self.get_stream_url_direct(video_id, websocket)

            if not hls_url:
                await websocket.send_json({
                    "type": "error",
                    "data": (
                        f"Could not access YouTube stream (ID: {video_id}).\n\n"
                        "Common causes:\n"
                        "- Stream has not started yet or has already ended\n"
                        "- Video is private, age-restricted, or members-only\n"
                        "- YouTube is rate-limiting the server\n\n"
                        "Tips:\n"
                        "- Confirm the stream is currently LIVE and public\n"
                        "- Test the URL in your browser first\n"
                        "- Try again in 30 seconds"
                    ),
                    "timestamp": datetime.now().isoformat()
                })
                return

            logger.info("Stream URL resolved")

            # ── 3. Audio capture + transcription loop ─────────────────────
            retry_count = 0
            detected_language = forced_language

            while self.connection_active and not self.stop_flag and retry_count < self.max_retries:
                try:
                    ffmpeg_cmd = [
                        "ffmpeg",
                        "-reconnect", "1",
                        "-reconnect_streamed", "1",
                        "-reconnect_delay_max", "5",
                        "-probesize", "32M",
                        "-analyzeduration", "10M",
                        "-i", hls_url,
                        "-vn", "-ac", "1", "-ar", str(self.sample_rate),
                        "-f", "s16le", "-loglevel", "error", "-"
                    ]

                    self.ffmpeg_process = subprocess.Popen(
                        ffmpeg_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        bufsize=0
                    )

                    asyncio.create_task(self.read_stderr(self.ffmpeg_process.stderr, "FFMPEG"))
                    asyncio.create_task(self.pipeline_monitor())

                    await websocket.send_json({
                        "type": "status",
                        "data": "Connected to stream. Processing audio...",
                        "timestamp": datetime.now().isoformat()
                    })

                    bps       = self.sample_rate * 2   # bytes per second (int16 mono)
                    chunk_sz  = bps * 1                # read 1 s at a time
                    threshold = bps * 5                # transcribe every 5 s
                    buffer    = bytearray()
                    last_log  = datetime.now()
                    logged_bytes = 0

                    await asyncio.sleep(2)  # let FFmpeg initialise

                    while self.connection_active and not self.stop_flag:
                        audio_data = await asyncio.get_event_loop().run_in_executor(
                            None, lambda: self.ffmpeg_process.stdout.read(chunk_sz)
                        )

                        if not audio_data:
                            if self.ffmpeg_process.poll() is not None:
                                logger.info("FFmpeg process ended")
                                break
                            await asyncio.sleep(0.05)
                            continue

                        self.last_audio_time = datetime.now()
                        logged_bytes += len(audio_data)
                        buffer.extend(audio_data)

                        if (datetime.now() - last_log).total_seconds() >= 5:
                            logger.info(f"Audio flow: {logged_bytes / 1024:.1f} KB received")
                            last_log = datetime.now()
                            logged_bytes = 0

                        if len(buffer) < threshold:
                            continue

                        # ── Transcribe ──────────────────────────────────
                        tmp_wav = None
                        try:
                            tmp_wav = await self._write_buffer_to_wav(bytes(buffer))
                            fw_model = app.state.fw_model

                            t0 = time.time()
                            segments, seg_info = fw_model.transcribe(
                                tmp_wav,
                                language=detected_language,
                                beam_size=5,
                                vad_filter=False,
                                condition_on_previous_text=False,
                                no_speech_threshold=0.4
                            )
                            logger.info(f"Transcription took {time.time() - t0:.2f}s")

                            transcript_text = " ".join(
                                seg.text.strip() for seg in segments if seg.text.strip()
                            )

                            if not transcript_text:
                                logger.warning("Empty transcript — skipping chunk")
                                continue

                            # ── Language detection (first chunk only) ───
                            if not detected_language:
                                if   re.search(r"[അ-ഹ]", transcript_text): detected_language = "ml"
                                elif re.search(r"[ऀ-ॿ]", transcript_text): detected_language = "hi"
                                elif re.search(r"[அ-ஹ]", transcript_text): detected_language = "ta"
                                elif re.search(r"[అ-హ]", transcript_text): detected_language = "te"
                                else:
                                    detected_language = getattr(seg_info, 'language', None) or "en"
                                self.detected_language = detected_language
                                logger.info(f"Detected language: {detected_language}")
                                await websocket.send_json({
                                    "type": "language_detected",
                                    "data": detected_language,
                                    "timestamp": datetime.now().isoformat()
                                })

                            # ── Garbage filter ──────────────────────────
                            garbage = ["film film", "20,000 years", "Thank you", "MBC News",
                                       "copyright", "subs by", "www.", ".com"]
                            if any(g.lower() in transcript_text.lower() for g in garbage):
                                continue
                            if is_repetition(transcript_text):
                                continue

                            # ── Summarise + translate ───────────────────
                            summary_base = summarize_text(transcript_text)
                            summaries: Dict[str, str] = {}
                            for lang in ["en", "ml", "hi", "ta", "te"]:
                                if lang == detected_language:
                                    summaries[lang] = summary_base
                                else:
                                    summaries[lang] = await translate_text(
                                        summary_base, detected_language, lang
                                    )

                            # ── Send to frontend ────────────────────────
                            if websocket.client_state.value == 3:
                                logger.warning("WebSocket disconnected")
                                break

                            await websocket.send_json({
                                "type": "transcription",
                                "data": transcript_text,
                                "language": detected_language,
                                "timestamp": datetime.now().isoformat()
                            })
                            await websocket.send_json({
                                "type": "summary",
                                "data": summaries,
                                "timestamp": datetime.now().isoformat()
                            })

                            self.transcript_buffer.append(transcript_text)
                            if len(self.transcript_buffer) > 100:
                                self.transcript_buffer.pop(0)

                        except Exception as e:
                            logger.error(f"Transcription block error: {e}", exc_info=True)
                        finally:
                            buffer.clear()
                            if tmp_wav and os.path.exists(tmp_wav):
                                try:
                                    os.remove(tmp_wav)
                                except Exception:
                                    pass

                    # Retry if pipeline ended unexpectedly
                    if self.connection_active and not self.stop_flag:
                        retry_count += 1
                        logger.warning(f"Pipeline ended unexpectedly (attempt {retry_count}/{self.max_retries})")
                        await asyncio.sleep(5 * retry_count)
                        continue

                except Exception as e:
                    logger.error(f"Processing error: {e}", exc_info=True)
                    retry_count += 1
                    await asyncio.sleep(5 * retry_count)
                    continue
                finally:
                    self.kill_processes()

            # ── 4. Final summary ──────────────────────────────────────────
            if self.transcript_buffer and self.connection_active:
                combined = " ".join(self.transcript_buffer)
                summaries = await self.generate_multilingual_summaries(combined, detected_language)
                try:
                    await websocket.send_json({
                        "type": "final_summary",
                        "data": summaries,
                        "timestamp": datetime.now().isoformat()
                    })
                except Exception as e:
                    logger.error(f"Error sending final summary: {e}")

        except Exception as e:
            logger.error(f"Critical stream error: {e}", exc_info=True)
            if self.connection_active:
                try:
                    await websocket.send_json({
                        "type": "error",
                        "data": f"Processing failed: {str(e)}",
                        "timestamp": datetime.now().isoformat()
                    })
                except Exception:
                    pass
        finally:
            self.stop_flag = True
            self.kill_processes()
            logger.info("Stream processing stopped")

    async def generate_multilingual_summaries(self, text: str, source_lang: Optional[str]) -> Dict[str, str]:
        if not source_lang or source_lang not in ['en', 'hi', 'ml', 'ta', 'te']:
            source_lang = 'en'
        summaries: Dict[str, str] = {source_lang: summarize_text(text)}
        for lang in ['en', 'hi', 'ml', 'ta', 'te']:
            if lang == source_lang:
                continue
            try:
                translated = await translate_text(text, source_lang, lang)
                summaries[lang] = summarize_text(translated)
            except Exception as e:
                logger.error(f"Summary translation to {lang} failed: {e}")
                summaries[lang] = "Translation unavailable"
        return summaries


# ============================================================================
# VIDEO FILE PROCESSING
# ============================================================================

async def process_video_file(
    video_path: str,
    translate: bool = True,
    summarize: bool = True,
    language: Optional[str] = None
):
    temp_audio = None
    try:
        logger.info(f"Processing video: {video_path}")
        if not os.path.exists(video_path):
            raise ValueError("Video file not found")

        duration = await asyncio.get_event_loop().run_in_executor(
            None, lambda: get_video_duration(video_path)
        )
        if duration > 600:
            raise ValueError("Video exceeds 10 minute limit")

        temp_audio = Path(tempfile.gettempdir()) / f"audio_{datetime.now().timestamp()}.wav"
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: subprocess.run(
                ["ffmpeg", "-y", "-i", video_path,
                 "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le",
                 "-f", "wav", "-vn", str(temp_audio)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
            )
        )

        audio, sr = sf.read(str(temp_audio))
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if audio is None or len(audio) == 0:
            raise ValueError("Audio loading failed")

        if len(audio) / sr > 300:
            raise HTTPException(status_code=400, detail="Videos are limited to 5 minutes on CPU")

        CHUNK_SECONDS = 30
        samples_per_chunk = int(CHUNK_SECONDS * sr)
        chunks = [audio[i:i + samples_per_chunk] for i in range(0, len(audio), samples_per_chunk)]
        logger.info(f"Audio split into {len(chunks)} chunks")

        fw_model = app.state.fw_model
        full_transcript = []

        for idx, chunk in enumerate(chunks):
            logger.info(f"Transcribing chunk {idx + 1}/{len(chunks)}")
            chunk = chunk.astype(np.float32)
            segments, _ = fw_model.transcribe(
                chunk, language=language, beam_size=5,
                vad_filter=False, condition_on_previous_text=False,
                no_speech_threshold=0.6
            )
            for seg in segments:
                if seg.text.strip():
                    full_transcript.append(seg.text.strip())

        transcript = " ".join(full_transcript)
        if not transcript:
            raise ValueError("Empty transcription result")

        if language:
            detected_language = language
        elif re.search(r"[അ-ഹ]", transcript): detected_language = "ml"
        elif re.search(r"[ऀ-ॿ]", transcript): detected_language = "hi"
        elif re.search(r"[அ-ஹ]", transcript): detected_language = "ta"
        elif re.search(r"[అ-హ]", transcript): detected_language = "te"
        else:                                   detected_language = "en"

        logger.info(f"Detected language: {detected_language}")
        response = {"transcript": transcript, "language": detected_language, "success": True}

        if translate:
            response["translations"] = {}
            for lang in ["en", "hi", "ml", "ta", "te"]:
                if lang == detected_language:
                    response["translations"][lang] = transcript
                else:
                    response["translations"][lang] = await translate_text(
                        transcript, detected_language, lang
                    )

        if summarize:
            response["summaries"] = {
                lang: summarize_text(text)
                for lang, text in response["translations"].items()
            }

        return response

    except Exception as e:
        logger.error(f"process_video_file failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_audio and Path(temp_audio).exists():
            try:
                os.remove(temp_audio)
            except Exception:
                pass


# ============================================================================
# FASTAPI APP
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up...")
    init_db()
    MODEL_SIZE = os.getenv("MODEL_SIZE", "small")
    fw_model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    app.state.fw_model = fw_model
    app.state.ws_processor = YouTubeLiveProcessor()
    logger.info("Application startup complete")
    yield
    logger.info("Shutting down...")
    shutdown_event.set()


app = FastAPI(
    title="Real-Time VTS API",
    description="Multilingual Video Transcript Summarizer — video upload, live streams, auth, download",
    version="3.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)


# ============================================================================
# AUTH ENDPOINTS
# ============================================================================

@app.post("/auth/register")
async def register(req: RegisterRequest):
    username = req.username.strip()
    password = req.password.strip()
    if not username or not password:
        raise HTTPException(status_code=400, detail="Username and password are required")
    if len(username) < 3:
        raise HTTPException(status_code=400, detail="Username must be at least 3 characters")
    if len(password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
    conn = get_db()
    try:
        if conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone():
            raise HTTPException(status_code=409, detail="Username already exists")
        conn.execute(
            "INSERT INTO users (username, password, created_at) VALUES (?, ?, ?)",
            (username, hash_password(password), datetime.now().isoformat())
        )
        conn.commit()
        token = create_session_token()
        user = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
        conn.execute(
            "INSERT INTO sessions (token, user_id, username, created_at) VALUES (?, ?, ?, ?)",
            (token, user["id"], username, datetime.now().isoformat())
        )
        conn.commit()
        logger.info(f"New user registered: {username}")
        return {"success": True, "token": token, "username": username, "message": "Registration successful"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")
    finally:
        conn.close()


@app.post("/auth/login")
async def login(req: LoginRequest):
    username = req.username.strip()
    password = req.password.strip()
    if not username or not password:
        raise HTTPException(status_code=400, detail="Username and password are required")
    conn = get_db()
    try:
        user = conn.execute(
            "SELECT * FROM users WHERE username = ? AND password = ?",
            (username, hash_password(password))
        ).fetchone()
        if not user:
            raise HTTPException(status_code=401, detail="Invalid username or password")
        token = create_session_token()
        conn.execute(
            "INSERT INTO sessions (token, user_id, username, created_at) VALUES (?, ?, ?, ?)",
            (token, user["id"], username, datetime.now().isoformat())
        )
        conn.execute(
            "UPDATE users SET last_login = ? WHERE id = ?",
            (datetime.now().isoformat(), user["id"])
        )
        conn.commit()
        logger.info(f"User logged in: {username}")
        return {"success": True, "token": token, "username": username, "message": "Login successful"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Login failed")
    finally:
        conn.close()


@app.post("/auth/logout")
async def logout(token: str):
    conn = get_db()
    try:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
        conn.commit()
        return {"success": True, "message": "Logged out successfully"}
    except Exception as e:
        logger.error(f"Logout error: {e}")
        raise HTTPException(status_code=500, detail="Logout failed")
    finally:
        conn.close()


@app.get("/auth/verify")
async def verify_session(token: str):
    user = verify_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    return {"valid": True, "username": user["username"]}


# ============================================================================
# DOWNLOAD ENDPOINT
# ============================================================================

@app.post("/download/text")
async def download_text(req: DownloadRequest):
    user = verify_token(req.token)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required to download")

    lang_names = {
        "english": "English", "hindi": "Hindi", "malayalam": "Malayalam",
        "tamil": "Tamil",     "telugu": "Telugu",
        "en": "English",      "hi": "Hindi", "ml": "Malayalam",
        "ta": "Tamil",        "te": "Telugu"
    }
    lang_name  = lang_names.get(req.language, req.language.capitalize())
    type_label = req.content_type.capitalize()
    timestamp  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    file_content = (
        f"VTS - Real-Time Multilingual Video Transcript Summarizer\n"
        f"{'=' * 60}\n"
        f"Downloaded by : {user['username']}\n"
        f"Date & Time   : {timestamp}\n"
        f"Language      : {lang_name}\n"
        f"Content Type  : {type_label}\n"
        f"{'=' * 60}\n\n"
        f"{req.content}\n\n"
        f"{'=' * 60}\n"
        f"Generated by VTS - Real-Time Multilingual Video Summarizer\n"
    )

    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO download_history (user_id, username, language, content_type, downloaded_at) VALUES (?, ?, ?, ?, ?)",
            (user["user_id"], user["username"], req.language, req.content_type, datetime.now().isoformat())
        )
        conn.commit()
    except Exception as e:
        logger.error(f"Download log error: {e}")
    finally:
        conn.close()

    filename = f"VTS_{type_label}_{req.language.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    return Response(
        content=file_content.encode("utf-8"),
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


# ============================================================================
# CORE API ENDPOINTS
# ============================================================================

@app.get("/")
async def health_check():
    return {
        "status": "running",
        "message": "Real-Time VTS API is operational",
        "version": "3.1.0",
        "supported_languages": ["en", "hi", "ml", "ta", "te"],
        "features": ["video_upload", "live_stream", "multi_language", "user_auth", "download"]
    }


@app.post("/upload/")
async def upload_video(
    file: UploadFile = File(...),
    translate: bool = True,
    summarize: bool = True,
    language: Optional[str] = None
):
    try:
        temp_dir = Path(tempfile.gettempdir()) / "video_uploads"
        temp_dir.mkdir(exist_ok=True)
        temp_video = temp_dir / f"upload_{datetime.now().timestamp()}{Path(file.filename).suffix}"

        with open(temp_video, "wb") as buf:
            buf.write(await file.read())

        try:
            result = await asyncio.wait_for(
                process_video_file(str(temp_video), translate, summarize, language),
                timeout=600
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Processing timed out after 10 minutes")
        finally:
            try:
                os.remove(temp_video)
            except Exception:
                pass

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"upload_video error: {e}")
        raise HTTPException(status_code=500, detail=str(e) or "Unknown error during processing")


@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    processor   = app.state.ws_processor
    stream_task = None

    try:
        async with processor.lock:
            processor.stop_flag = True
            processor.kill_processes()
            while processor.active_websocket is not None:
                await asyncio.sleep(0.1)
            processor.stop_flag         = False
            processor.current_retries   = 0
            processor.transcript_buffer = []
            processor.summary_buffer    = []
            processor.final_summaries   = {'en': '', 'hi': '', 'ml': '', 'ta': '', 'te': ''}
            processor.active_websocket  = websocket
            processor.connection_active = True

        logger.info("WebSocket connection established")

        while processor.connection_active:
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=300)

                if data.get('action') == 'stop':
                    logger.info("Stop command received")
                    async with processor.lock:
                        processor.stop_flag         = True
                        processor.connection_active = False

                    if stream_task and not stream_task.done():
                        stream_task.cancel()
                        try:
                            await stream_task
                        except asyncio.CancelledError:
                            pass

                    transcripts  = processor.transcript_buffer.copy()
                    current_lang = processor.detected_language or 'en'

                    if transcripts:
                        try:
                            combined = " ".join(transcripts)
                            summaries = await processor.generate_multilingual_summaries(combined, current_lang)
                        except Exception as e:
                            logger.error(f"Final summary error: {e}")
                            summaries = {l: "Summary generation failed" for l in ["en", "hi", "ml", "ta", "te"]}
                    else:
                        summaries = {l: "No transcripts collected" for l in ["en", "hi", "ml", "ta", "te"]}

                    try:
                        await websocket.send_json({
                            "type": "final_summary",
                            "data": summaries,
                            "timestamp": datetime.now().isoformat()
                        })
                    except Exception:
                        pass

                    await asyncio.sleep(2.0)
                    await websocket.send_json({
                        "type": "stopped",
                        "data": "Processing stopped",
                        "timestamp": datetime.now().isoformat()
                    })
                    break

                elif data.get('action') == 'start':
                    url = data.get('url', '').strip()
                    if not url:
                        await websocket.send_json({"type": "error", "data": "URL required"})
                        continue
                    if "youtube.com" not in url and "youtu.be" not in url:
                        await websocket.send_json({"type": "error", "data": "Only YouTube URLs are supported"})
                        continue
                    if stream_task and not stream_task.done():
                        await websocket.send_json({"type": "error", "data": "Stream already running"})
                        continue
                    lang = data.get('language')
                    logger.info(f"Starting stream: {url}")
                    stream_task = asyncio.create_task(
                        processor.process_stream(url, websocket, forced_language=lang)
                    )

                else:
                    # Legacy format: {"url": "..."}
                    url = data.get('url', '').strip()
                    if url:
                        logger.warning("Using legacy message format")
                        if "youtube.com" not in url and "youtu.be" not in url:
                            await websocket.send_json({"type": "error", "data": "Only YouTube URLs are supported"})
                            continue
                        async with processor.lock:
                            processor.stop_flag         = False
                            processor.connection_active = True
                            processor.transcript_buffer = []
                            processor.detected_language = None
                        stream_task = asyncio.create_task(processor.process_stream(url, websocket))

            except asyncio.TimeoutError:
                logger.warning("WebSocket idle timeout (300s)")
                continue
            except WebSocketDisconnect:
                logger.info("Client disconnected")
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}", exc_info=True)
                try:
                    await websocket.send_json({"type": "error", "data": str(e)})
                except Exception:
                    pass
                break

    finally:
        async with processor.lock:
            processor.stop_flag         = True
            processor.connection_active = False
            processor.kill_processes()
            processor.active_websocket  = None
        try:
            if stream_task:
                stream_task.cancel()
        except Exception:
            pass
        try:
            await websocket.close()
        except Exception:
            pass
        logger.info("WebSocket connection closed")


@app.post("/stop_stream/")
async def stop_stream(request: StopStreamRequest):
    try:
        processor = app.state.ws_processor
        async with processor.lock:
            processor.stop_flag         = True
            processor.connection_active = False
            processor.kill_processes()
        all_text = " ".join(t.text for t in request.transcriptions if t.text.strip())
        if not all_text.strip():
            return {
                "success": False,
                "message": "No transcriptions available",
                "summaries": {l: "No transcriptions available" for l in ["en", "hi", "ml", "ta", "te"]}
            }
        summaries = await processor.generate_multilingual_summaries(all_text, request.language or "en")
        return {"success": True, "message": "Stream stopped", "summaries": summaries}
    except Exception as e:
        logger.error(f"stop_stream error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate_summary/")
async def generate_summary(request: GenerateSummaryRequest):
    try:
        processor = app.state.ws_processor
        all_text  = " ".join(t.text for t in request.transcriptions if t.text.strip())
        if not all_text.strip():
            raise HTTPException(status_code=400, detail="No transcriptions available")
        summaries = await processor.generate_multilingual_summaries(all_text, request.language or "en")
        return {"success": True, "summaries": summaries}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"generate_summary error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    signal.signal(signal.SIGINT,  lambda s, f: shutdown_event.set())
    signal.signal(signal.SIGTERM, lambda s, f: shutdown_event.set())

    logger.info("Starting Real-Time VTS API v3.1.0")
    logger.info("Auth    : SQLite login/register")
    logger.info("Download: .txt export per language")
    logger.info("Streams : 6-strategy yt-dlp extractor")
    logger.info("Server  : http://localhost:8000")
    logger.info("Docs    : http://localhost:8000/docs")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        timeout_keep_alive=300,
        ws_ping_interval=20,
        ws_ping_timeout=20
    )
