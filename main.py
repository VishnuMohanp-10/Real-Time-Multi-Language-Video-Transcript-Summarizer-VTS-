import os
import tempfile
import whisper
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yt_dlp as youtube_dl
from faster_whisper import WhisperModel
import torch
import logging
import asyncio
import subprocess
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List
from deep_translator import GoogleTranslator  # ✅ FIXED: Using deep-translator instead of googletrans
import re
import signal
from contextlib import asynccontextmanager
import random
import soundfile as sf
import psutil
from pydantic import BaseModel
import wave
import shutil
import httpx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# Global flag for shutdown
shutdown_event = asyncio.Event()


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

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
    """
    Simple sentence splitter that works better for Indic scripts:
    splits on ., ?, ! and Devanagari danda (।)
    """
    sentences = re.split(r'(?<=[\.!\?।])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def is_repetition(text: str) -> bool:
    """Check if text contains repetitive patterns"""
    # 1. Check for simple word repetition (e.g., "word word word")
    tokens = text.split()
    if len(tokens) >= 3:
        unique_tokens = set(tokens)
        if len(unique_tokens) / len(tokens) < 0.5:  # If >50% words are same
            return True

    # 2. Check for character repetition (e.g., "nnnnnnnnnnnn")
    if len(text) > 10 and len(set(text)) < 5:
        return True
        
    return False


def summarize_text(text: str, max_sentences: int = 3, max_chars: int = 500) -> str:
    """Generate a summary from text"""
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
    """Get video duration using ffprobe"""
    try:
        result = subprocess.run([
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if result.returncode == 0 and result.stdout:
            return float(result.stdout.strip())
        else:
            raise ValueError("Could not determine video duration")
    except Exception as e:
        logger.error(f"Error getting video duration: {str(e)}")
        raise ValueError("Could not determine video duration")


def extract_hls_url(info):
    """Extract HLS URL from yt-dlp info dict"""
    if not info or not isinstance(info, dict):
        return None
        
    formats = info.get('formats', [])
    # Try to find HLS formats
    hls_formats = [f for f in formats if f.get('protocol') == 'm3u8' or '.m3u8' in f.get('url', '')]
    
    if hls_formats:
        # Prefer audio-only HLS
        audio_hls = [f for f in hls_formats if f.get('acodec') != 'none' and f.get('vcodec') == 'none']
        if audio_hls:
            return sorted(audio_hls, key=lambda f: f.get('tbr', 0), reverse=True)[0]['url']
        
        # Return any HLS format
        return sorted(hls_formats, key=lambda f: f.get('tbr', 0), reverse=True)[0]['url']
    
    # If no HLS, try to find any working format
    working_formats = [f for f in formats if f.get('url')]
    if working_formats:
        return sorted(working_formats, key=lambda f: f.get('tbr', 0), reverse=True)[0]['url']
    
    return None


async def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    """
    Translate text using deep-translator (more stable than googletrans)
    """
    try:
        translator = GoogleTranslator(source=source_lang, target=target_lang)
        # Run in thread to avoid blocking
        result = await asyncio.to_thread(translator.translate, text)
        return result
    except Exception as e:
        logger.error(f"Translation from {source_lang} to {target_lang} failed: {e}")
        return f"Translation unavailable: {str(e)}"


# ============================================================================
# YOUTUBE LIVE PROCESSOR CLASS
# ============================================================================

class YouTubeLiveProcessor:
    def __init__(self):
        self.stop_flag = False
        self.cookies_file = "cookies.txt"
        self.live_sentence_buffer = ""
        self.sample_rate = 16000
        self.detected_language = None
        self.live_transcript = ""
        self.max_retries = 5
        self.current_retries = 0
        self.transcript_buffer = []
        self.summary_buffer = []
        self.final_summaries = {'en': '', 'hi': '', 'ml': '', 'ta': '', 'te': ''}
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        ]
        self.ffmpeg_process = None
        self.ydl_process = None
        self.active_websocket = None
        self.current_url = None
        self.last_audio_time = datetime.now()
        self.connection_active = False
        self.lock = asyncio.Lock()
        self.audio_stall_timeout = 90

    def kill_processes(self):
        """Forcefully kill all running processes"""
        try:
            if self.ffmpeg_process:
                try:
                    self.ffmpeg_process.kill()
                except Exception:
                    pass
                self.ffmpeg_process = None
            if self.ydl_process:
                try:
                    self.ydl_process.kill()
                except Exception:
                    pass
                self.ydl_process = None

            for proc in psutil.process_iter(['name']):
                try:
                    if proc.info['name'] in ['ffmpeg', 'yt-dlp', 'yt-dlp.exe', 'ffmpeg.exe']:
                        proc.kill()
                except Exception:
                    continue
        except Exception as e:
            logger.error(f"Error killing processes: {e}")

    async def check_stream_health(self):
        """Check if the stream is still live"""
        try:
            ydl = youtube_dl.YoutubeDL({
                'quiet': True,
                'socket_timeout': 10,
                'user_agent': random.choice(self.user_agents)
            })
            info = await asyncio.to_thread(ydl.extract_info, self.current_url, download=False)
            return info.get('is_live', False)
        except Exception as e:
            logger.error(f"Stream health check failed: {str(e)}")
            return False

    async def pipeline_monitor(self):
        """Monitor the health of the processing pipeline"""
        await asyncio.sleep(15)
        
        while self.connection_active and not self.stop_flag and not shutdown_event.is_set():
            async with self.lock:
                # Check process status
                if (self.ffmpeg_process and self.ffmpeg_process.poll() is not None) or \
                   (self.ydl_process and self.ydl_process.poll() is not None):
                    logger.error("Subprocess died unexpectedly")
                    self.kill_processes()
                    break

                # Check for audio stall
                if (datetime.now() - self.last_audio_time).total_seconds() > self.audio_stall_timeout:
                    logger.error("Audio stall detected (no data for 90 seconds)")
                    self.kill_processes()
                    break

            await asyncio.sleep(15)

    async def read_stderr(self, pipe, prefix):
        """Read stderr output with enhanced error handling"""
        error_buffer = []
        loop = asyncio.get_event_loop()
        while self.connection_active and not self.stop_flag:
            try:
                line = await loop.run_in_executor(None, pipe.readline)
                if line:
                    try:
                        decoded = line.decode(errors='ignore').strip()
                    except Exception:
                        decoded = str(line)
                    error_buffer.append(decoded)
                    logger.debug(f"{prefix}: {decoded}")
                else:
                    if error_buffer:
                        logger.error(f"{prefix} ERROR OUTPUT:\n" + "\n".join(error_buffer[-10:]))
                    break
            except Exception as e:
                logger.error(f"Error reading stderr: {str(e)}")
                break

    async def get_stream_url_direct(self, video_id: str, websocket: WebSocket) -> Optional[str]:
        """Try to get stream URL using direct methods (bypass yt-dlp)"""
        try:
            # Method 1: Try to construct a direct HLS URL pattern
            potential_urls = [
                f"https://www.youtube.com/watch?v={video_id}",
                f"https://www.youtube.com/embed/{video_id}",
                f"https://www.youtube.com/live/{video_id}",
            ]
            
            # Try each potential URL with FFmpeg directly
            for test_url in potential_urls:
                try:
                    # Test if FFmpeg can access the URL directly
                    test_cmd = [
                        'ffmpeg',
                        '-i', test_url,
                        '-t', '5',  # Test for 5 seconds only
                        '-f', 'null',
                        '-'
                    ]
                    
                    result = subprocess.run(
                        test_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=10
                    )
                    
                    if result.returncode == 0:
                        logger.info(f"✅ Direct URL works: {test_url}")
                        return test_url
                            
                except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
                    continue
                except Exception as e:
                    logger.warning(f"Test failed for {test_url}: {str(e)}")
                    continue
            
            # Method 2: Try to extract from embed page
            try:
                embed_url = f"https://www.youtube.com/embed/{video_id}"
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Referer': 'https://www.youtube.com/',
                }
                
                async with httpx.AsyncClient() as client:
                    response = await client.get(embed_url, headers=headers, timeout=30)
                    if response.status_code == 200:
                        content = response.text
                        # Look for stream URLs in the content
                        patterns = [
                            r'"hlsManifestUrl":"([^"]+)"',
                            r'https://[^"]*\.m3u8[^"]*',
                            r'manifest.*?https://[^"]*',
                        ]
                        
                        for pattern in patterns:
                            matches = re.findall(pattern, content)
                            for match in matches:
                                if 'm3u8' in match:
                                    hls_url = match.replace('\\u0026', '&').replace('\\/', '/')
                                    logger.info(f"Found HLS URL in embed page: {hls_url[:100]}...")
                                    return hls_url
            except Exception as e:
                logger.warning(f"Embed page method failed: {str(e)}")
                
        except Exception as e:
            logger.error(f"Direct URL extraction failed: {str(e)}")
        
        return None

    def generate_detailed_error_message(self, video_id: str) -> str:
        """Generate detailed error message with troubleshooting steps"""
        return f"""
🚫 Could not access YouTube live stream

📋 Stream ID: {video_id}
🔗 Test URL: https://www.youtube.com/watch?v={video_id}

🔍 Possible reasons:
• The stream is private, members-only, or requires subscription
• The stream has age restrictions
• The stream is not available in your region
• YouTube has blocked automated access
• The stream may have ended or not started yet

💡 Solutions to try:
1. 📋 Create a cookies.txt file for authentication
2. 🌐 Try a different live stream URL
3. 🔄 Update yt-dlp: pip install -U yt-dlp
4. 🎯 Test with a public stream first
        """

    async def _write_buffer_to_wav(self, buffer_bytes: bytes) -> str:
        """
        Write raw PCM int16 samples (mono, 16kHz) to a temporary WAV file
        """
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.close()
        try:
            with wave.open(tmp.name, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # int16 => 2 bytes
                wf.setframerate(self.sample_rate)
                wf.writeframes(buffer_bytes)
            return tmp.name
        except Exception as e:
            logger.error(f"Failed to write WAV temp file: {e}")
            try:
                os.unlink(tmp.name)
            except Exception:
                pass
            raise

    async def process_stream(self, url: str, websocket: WebSocket, forced_language: Optional[str] = None):
        """Process YouTube live stream with buffering and multi-language support"""
        self.current_url = url
        logger.info(f"🚀 Starting stream processing for URL: {url}")

        try:
            # Extract video ID from URL
            video_id = None
            patterns = [
                r'youtube\.com/live/([a-zA-Z0-9_-]+)',
                r'youtu\.be/([a-zA-Z0-9_-]+)',
                r'v=([a-zA-Z0-9_-]+)',
                r'youtube\.com/watch/([a-zA-Z0-9_-]+)'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, url)
                if match:
                    video_id = match.group(1)
                    break
            
            if not video_id:
                await websocket.send_json({
                    "type": "error",
                    "data": "Invalid YouTube URL format",
                    "timestamp": datetime.now().isoformat()
                })
                return

            logger.info(f"📋 Extracted video ID: {video_id}")

            # Try to get stream URL using direct methods
            hls_url = await self.get_stream_url_direct(video_id, websocket)
            
            if not hls_url:
                # Fallback: try yt-dlp
                try:
                    ydl_opts = {
                        'quiet': True,
                        'socket_timeout': 60,
                        'user_agent': random.choice(self.user_agents),
                        'cookiefile': self.cookies_file if os.path.exists(self.cookies_file) else None,
                        'extractor_args': {
                            'youtube': {
                                'player_client': ['android', 'web'],
                                'player_skip': ['configs', 'webpage'],
                            }
                        },
                    }
                    
                    ydl = youtube_dl.YoutubeDL(ydl_opts)
                    info = await asyncio.to_thread(ydl.extract_info, f"https://www.youtube.com/watch?v={video_id}", download=False, process=False)
                    hls_url = extract_hls_url(info)
                    
                except Exception as e:
                    logger.error(f"yt-dlp fallback failed: {str(e)}")

            if not hls_url:
                error_msg = self.generate_detailed_error_message(video_id)
                await websocket.send_json({
                    "type": "error",
                    "data": error_msg,
                    "timestamp": datetime.now().isoformat()
                })
                return

            logger.info(f"✅ Successfully extracted stream URL")

            retry_count = 0
            detected_language = forced_language

            while self.connection_active and not self.stop_flag and retry_count < self.max_retries:
                try:
                    # Start FFmpeg with HLS URL
                    ffmpeg_cmd = [
                        "ffmpeg",
                        "-reconnect", "1",
                        "-reconnect_streamed", "1",
                        "-reconnect_delay_max", "5",
                        "-probesize", "32M",
                        "-analyzeduration", "10M",
                        "-i", hls_url,
                        "-vn",
                        "-ac", "1",
                        "-ar", str(self.sample_rate),
                        "-f", "s16le",
                        "-loglevel", "error",
                        "-"
                    ]

                    self.ffmpeg_process = subprocess.Popen(
                        ffmpeg_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        bufsize=0
                    )

                    # Start monitoring tasks
                    asyncio.create_task(self.read_stderr(self.ffmpeg_process.stderr, "FFMPEG"))
                    asyncio.create_task(self.pipeline_monitor())
                    
                    await websocket.send_json({
                        "type": "status",
                        "data": "Connected to stream. Processing audio...",
                        "timestamp": datetime.now().isoformat()
                    })

                    # Audio processing parameters
                    bytes_per_second = self.sample_rate * 2  # 2 bytes per sample (int16)
                    chunk_size = bytes_per_second * 1  # read 1 second at a time
                    buffer = bytearray()
                    buffer_threshold_bytes = bytes_per_second * 5  # 5 seconds buffer
                    last_status_time = datetime.now()
                    bytes_processed = 0

                    await asyncio.sleep(2)  # Let FFmpeg initialize

                    # Main audio read/process loop
                    while self.connection_active and not self.stop_flag:
                        # Read a chunk
                        audio_data = await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda: self.ffmpeg_process.stdout.read(chunk_size)
                        )

                        if not audio_data:
                            if self.ffmpeg_process.poll() is not None:
                                logger.info("FFmpeg process ended")
                                break
                            await asyncio.sleep(0.05)
                            continue

                        self.last_audio_time = datetime.now()
                        bytes_processed += len(audio_data)
                        buffer.extend(audio_data)

                        # Periodic status every 5s
                        if (datetime.now() - last_status_time).total_seconds() >= 5:
                            logger.info(f"🔊 Audio flow: {bytes_processed/1024:.1f}KB received")
                            last_status_time = datetime.now()
                            bytes_processed = 0
                        
                        # Check if buffer reached threshold to transcribe
                        if len(buffer) >= buffer_threshold_bytes:
                            tmp_wav = None

                            try:
                                # Write buffer to WAV
                                tmp_wav = await self._write_buffer_to_wav(bytes(buffer))

                                # Whisper transcription using faster-whisper
                                fw_model = app.state.fw_model

                                segments, info = fw_model.transcribe(
                                    tmp_wav,
                                    language=detected_language,
                                    beam_size=5,
                                    vad_filter=False,  # 🔥 FIXED: Disabled aggressive VAD for live streams
                                    condition_on_previous_text=False,
                                    no_speech_threshold=0.4  # Higher threshold = less sensitive to noise
                                )

                                # Collect transcript from segments
                                transcript_text = ""
                                segment_count = 0
                                for segment in segments:
                                    segment_count += 1
                                    if segment.text.strip():
                                        transcript_text += segment.text.strip() + " "
                                
                                transcript_text = transcript_text.strip()


                                # 🔍 DEBUG: Log what Whisper returned
                                logger.info(f"📊 Whisper: {segment_count} segments, {len(transcript_text)} chars")
                                if transcript_text:
                                    logger.info(f"📝 Transcript: {transcript_text[:100]}...")
                                else:
                                    logger.warning(f"⚠️ Empty transcript from {segment_count} segments")


                                if not transcript_text:
                                    logger.warning("⚠️ Skipping empty transcript")
                                    buffer.clear()
                                    if tmp_wav and os.path.exists(tmp_wav):
                                        try:
                                            os.remove(tmp_wav)
                                        except Exception:
                                            pass
                                    continue  # Don't raise error, just skip```

                                if not transcript_text:
                                    raise ValueError("Empty transcript")

                                # Language detection (if not forced)
                                if not detected_language:
                                    if re.search(r"[അ-ഹ]", transcript_text):
                                        detected_language = "ml"
                                    elif re.search(r"[ऀ-ॿ]", transcript_text):
                                        detected_language = "hi"
                                    elif re.search(r"[அ-ஹ]", transcript_text):
                                        detected_language = "ta"
                                    elif re.search(r"[అ-హ]", transcript_text):
                                        detected_language = "te"
                                    else:
                                        detected_language = info.language if hasattr(info, 'language') else "en"
                                    
                                    self.detected_language = detected_language
                                    logger.info(f"🔍 Detected language: {detected_language}")
                                    
                                    await websocket.send_json({
                                        "type": "language_detected",
                                        "data": detected_language,
                                        "timestamp": datetime.now().isoformat()
                                    })

                                # Filter garbage
                                garbage_phrases = [
                                    "film film", "20,000 years", "Thank you", "MBC News", 
                                    "copyright", "subs by", "www.", ".com"
                                ]

                                if any(garbage.lower() in transcript_text.lower() for garbage in garbage_phrases):
                                    buffer.clear()
                                    if tmp_wav and os.path.exists(tmp_wav):
                                        try:
                                            os.remove(tmp_wav)
                                        except Exception:
                                            pass
                                    continue
                                
                                if is_repetition(transcript_text):
                                    buffer.clear()
                                    if tmp_wav and os.path.exists(tmp_wav):
                                        try:
                                            os.remove(tmp_wav)
                                        except Exception:
                                            pass
                                    continue

                                # Summarize in same language
                                summary_base = summarize_text(transcript_text)

                                # Translate summary to all languages using deep-translator
                                summaries = {}
                                
                                for lang in ["en", "ml", "hi", "ta", "te"]:
                                    if lang == detected_language:
                                        summaries[lang] = summary_base
                                    else:
                                        try:
                                            summaries[lang] = await translate_text(summary_base, detected_language, lang)
                                        except Exception as e:
                                            logger.error(f"Translation to {lang} failed: {e}")
                                            summaries[lang] = "Translation unavailable"

                                # Send over WebSocket
                                try:
                                    # Check WebSocket is still open
                                    if websocket.client_state.value == 3:  # DISCONNECTED
                                        logger.warning("⚠️ WebSocket disconnected, stopping")
                                        break
    
                                    await websocket.send_json({
                                        "type": "transcription",
                                        "data": transcript_text,
                                        "language": detected_language,
                                        "timestamp": datetime.now().isoformat()
                                    })
                                    logger.info(f"✅ Sent transcription ({len(transcript_text)} chars)")

                                    await websocket.send_json({
                                        "type": "summary",
                                        "data": summaries,
                                        "timestamp": datetime.now().isoformat()
                                    })
                                    logger.info(f"✅ Sent summaries (5 languages)")

                                except Exception as e:
                                    logger.error(f"WebSocket send failed: {e}")
                                    break # Exit if we can't send 

                                # Store for final summary
                                self.transcript_buffer.append(transcript_text)
                                if len(self.transcript_buffer) > 100:
                                    self.transcript_buffer.pop(0)

                            except Exception as e:
                                logger.error(f"Transcription block failed: {e}", exc_info=True)

                            finally:
                                buffer.clear()
                                if tmp_wav and os.path.exists(tmp_wav):
                                    try:
                                        os.remove(tmp_wav)
                                    except Exception:
                                        pass

                    # If loop ended but we were not asked to stop, attempt to retry
                    if self.connection_active and not self.stop_flag:
                        logger.warning(f"⚠️ Pipeline ended unexpectedly (attempt {retry_count + 1}/{self.max_retries})")
                        retry_count += 1
                        await asyncio.sleep(5 * (retry_count + 1))
                        continue

                except Exception as e:
                    logger.error(f"Processing error: {str(e)}", exc_info=True)
                    retry_count += 1
                    await asyncio.sleep(5 * (retry_count + 1))
                    continue
 
                finally:
                    self.kill_processes()

            # Generate final summary
            if self.transcript_buffer and self.connection_active:
                combined_text = " ".join(self.transcript_buffer)
                summaries = await self.generate_multilingual_summaries(combined_text, detected_language)
                try:
                    await websocket.send_json({
                        "type": "final_summary",
                        "data": summaries,
                        "timestamp": datetime.now().isoformat()
                    })
                except Exception as e:
                    logger.error(f"Error sending final summary: {str(e)}")

        except Exception as e:
            logger.error(f"🚨 Critical error in stream processing: {str(e)}", exc_info=True)
            if self.connection_active:
                try:
                    await websocket.send_json({
                        "type": "error",
                        "data": f"Processing failed: {str(e)}",
                        "timestamp": datetime.now().isoformat()
                    })
                except Exception as ex:
                    logger.error(f"Error sending error message: {ex}")

        finally:
            self.stop_flag = True
            self.kill_processes()
            logger.info("🛑 Stream processing stopped")

    async def generate_multilingual_summaries(self, text: str, source_lang: Optional[str]) -> Dict[str, str]:
        """Generate summaries in all supported languages using deep-translator"""
        summaries = {}

        if not source_lang or source_lang not in ['en', 'hi', 'ml', 'ta', 'te']:
            source_lang = 'en'

        try:
            summaries[source_lang] = summarize_text(text)

            target_langs = ['en', 'hi', 'ml', 'ta', 'te']
            if source_lang in target_langs:
                target_langs.remove(source_lang)

            for lang in target_langs:
                try:
                    translated = await translate_text(text, source_lang, lang)
                    summaries[lang] = summarize_text(translated)
                except Exception as e:
                    logger.error(f"Translation to {lang} failed: {str(e)}")
                    summaries[lang] = "Translation unavailable"

        except Exception as e:
            logger.error(f"Error generating summaries: {str(e)}")
            summaries = {'en': "Summary failed", 'hi': "Summary failed", 'ml': "Summary failed", 'ta': "Summary failed", 'te': "Summary failed"}
            summaries[source_lang] = summarize_text(text)

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
    """Process uploaded video file with multi-language transcription and translation"""
    temp_audio = None
    try:
        logger.info(f"Processing video: {video_path}")
        if not os.path.exists(video_path):
            raise ValueError("Video file not found")

        # 1. Check VIDEO duration (hard limit)
        duration = await asyncio.get_event_loop().run_in_executor(
            None, lambda: get_video_duration(video_path)
        )
        if duration > 600:
            raise ValueError("Video exceeds 10 minute limit")

        # 2. Extract audio
        temp_audio = Path(tempfile.gettempdir()) / f"audio_{datetime.now().timestamp()}.wav"
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: subprocess.run(
                [
                    "ffmpeg", "-y", "-i", video_path,
                    "-ac", "1", "-ar", "16000",
                    "-acodec", "pcm_s16le",
                    "-f", "wav", "-vn",
                    str(temp_audio)
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
        )

        # 3. Load audio
        audio, sr = sf.read(str(temp_audio))

        # Ensure mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if audio is None or len(audio) == 0:
            raise ValueError("Audio loading failed")

        # 4. CPU safety limit
        MAX_AUDIO_SECONDS = 300  # 5 minutes on CPU
        audio_duration = len(audio) / sr

        if audio_duration > MAX_AUDIO_SECONDS:
            raise HTTPException(
                status_code=400,
                detail="Videos are limited to 5 minutes on CPU"
            )

        # 5. Chunking for large files
        CHUNK_SECONDS = 30
        samples_per_chunk = CHUNK_SECONDS * sr

        chunks = [
            audio[i:i + samples_per_chunk]
            for i in range(0, len(audio), samples_per_chunk)
        ]

        logger.info(f"Audio split into {len(chunks)} chunks")

        fw_model = app.state.fw_model
        full_transcript = []

        for idx, chunk in enumerate(chunks):
            logger.info(f"Transcribing chunk {idx + 1}/{len(chunks)}")

            chunk = chunk.astype(np.float32)

            segments, info = fw_model.transcribe(
                chunk,
                language=language,
                beam_size=5,
                vad_filter=False,  # 🔥 FIXED: Disabled for consistency
                condition_on_previous_text=False,
                no_speech_threshold=0.6
            )

            for segment in segments:
                if segment.text.strip():
                    full_transcript.append(segment.text.strip())

        transcript = " ".join(full_transcript)

        if not transcript:
            raise ValueError("Empty transcription result")

        # 6. Language detection
        if language:
            detected_language = language
        elif re.search(r"[അ-ഹ]", transcript):
            detected_language = "ml"
        elif re.search(r"[ऀ-ॿ]", transcript):
            detected_language = "hi"
        elif re.search(r"[அ-ஹ]", transcript):
            detected_language = "ta"
        elif re.search(r"[అ-హ]", transcript):
            detected_language = "te"
        else:
            detected_language = "en"

        logger.info(f"Detected language: {detected_language}")

        response = {
            "transcript": transcript,
            "language": detected_language,
            "success": True
        }

        # 7. Translation using deep-translator
        if translate:
            response["translations"] = {}
            for lang in ["en", "hi", "ml", "ta", "te"]:
                if lang == detected_language:
                    response["translations"][lang] = transcript
                    continue
                try:
                    response["translations"][lang] = await translate_text(transcript, detected_language, lang)
                except Exception as e:
                    logger.error(f"Translation to {lang} failed: {e}")
                    response["translations"][lang] = "Translation unavailable"

        # 8. Summarization
        if summarize:
            response["summaries"] = {}
            for lang, text in response["translations"].items():
                response["summaries"][lang] = summarize_text(text)

        return response

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if temp_audio and temp_audio.exists():
            try:
                os.remove(temp_audio)
            except Exception:
                pass


# ============================================================================
# FASTAPI APPLICATION SETUP
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown"""
    logger.info("Starting up...")

    # Initialize faster-whisper model
    fw_model = WhisperModel(
        "small",
        device="cpu",
        compute_type="int8"
    )

    app.state.fw_model = fw_model
    app.state.ws_processor = YouTubeLiveProcessor()

    logger.info("✅ Application startup complete")

    yield

    # Shutdown cleanup
    logger.info("Shutting down...")
    shutdown_event.set()


app = FastAPI(
    title="Multilingual Video Summarizer API",
    description="API for processing videos and live streams with multi-language support",
    version="2.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def health_check():
    status = {
        "status": "running",
        "message": "Multilingual Video Summarizer API is operational",
        "supported_languages": ["en", "hi", "ml", "ta", "te"],
        "translation_available": True,
        "max_video_duration": "10 minutes",
        "features": ["video_upload", "live_stream", "multi_language"]
    }
    return status


@app.post("/upload/")
async def upload_video(
    file: UploadFile = File(...),
    translate: bool = True,
    summarize: bool = True,
    language: Optional[str] = None
):
    """Upload and process a video file"""
    try:
        temp_dir = Path(tempfile.gettempdir()) / "video_uploads"
        temp_dir.mkdir(exist_ok=True)

        temp_video = temp_dir / f"upload_{datetime.now().timestamp()}{Path(file.filename).suffix}"
        with open(temp_video, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        try:
            result = await asyncio.wait_for(
                process_video_file(
                    str(temp_video),
                    translate,
                    summarize,
                    language
                ),
                timeout=600
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Processing timed out after 10 minutes")

        try:
            os.remove(temp_video)
        except Exception:
            pass

        return result
    except Exception as e:
        logger.error(f"Video upload error: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(
            status_code=500,
            detail=str(e) if str(e) else "An unknown error occurred during processing"
        )


@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for live stream processing"""
    await websocket.accept()
    processor = app.state.ws_processor

    stream_task = None

    try:
        async with processor.lock:
            # Ensure no other processor is active
            processor.stop_flag = True
            processor.kill_processes()
            while processor.active_websocket is not None:
                await asyncio.sleep(0.1)

            processor.stop_flag = False
            processor.current_retries = 0
            processor.transcript_buffer = []
            processor.summary_buffer = []
            processor.final_summaries = {'en': '', 'hi': '', 'ml': '', 'ta': '', 'te': ''}
            processor.active_websocket = websocket
            processor.connection_active = True

        logger.info("✅ WebSocket connection established")

        while processor.connection_active:
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=300)

                # STOP COMMAND
                if data.get('action') == 'stop':
                    logger.info("Stop command received")

                    #1. Set stop Flags
                    async with processor.lock:
                        processor.stop_flag = True
                        processor.connection_active = False

                    #2. Cancle stream task properly
                    if stream_task and not stream_task.done():
                        stream_task.cancel()
                        try:
                            await stream_task
                        except asyncio.CancelledError:
                            logger.info("Stream task cancelled successfully")

                    # Generate final summary
                    transcripts = processor.transcript_buffer.copy()
                    current_lang = processor.detected_language if processor.detected_language else 'en'
                    
                    summaries = None
                    if transcripts:
                        try:
                            combined_text = " ".join(transcripts)
                            summaries = await processor.generate_multilingual_summaries(combined_text, current_lang)
                            logger.info("✅ Final summary generated")
                        except Exception as e:
                            logger.error(f"Summary generation failed: {str(e)}")
                            summaries = {lang: "Summary generation failed" for lang in ["en", "hi", "ml", "ta", "te"]}
                    else:
                        summaries = {lang: "No transcripts collected" for lang in ["en", "hi", "ml", "ta", "te"]}

                    # Send final summary
                    if summaries:
                        try:
                            await websocket.send_json({
                                "type": "final_summary",
                                "data": summaries,
                                "timestamp": datetime.now().isoformat()
                            })
                        except Exception as e:
                            logger.warning(f"Could not send final summary: {str(e)}")
                    
                    await asyncio.sleep(2.0)

                    await websocket.send_json({
                        "type": "stopped",
                        "data": "Processing stopped",
                        "timestamp": datetime.now().isoformat()
                    })
                    break

                # START STREAM
                elif data.get('action') == 'start':
                    url = data.get('url')
                    if not url:
                        await websocket.send_json({"type": "error", "data": "URL required"})
                        continue

                    if "youtube.com" not in url and "youtu.be" not in url:
                        await websocket.send_json({"type": "error", "data": "Only YouTube URLs supported"})
                        continue

                    if stream_task and not stream_task.done():
                        await websocket.send_json({
                            "type": "error",
                            "data": "Stream already running"
                        })
                        continue

                    lang = data.get('language')  # optional forced language

                    logger.info(f"Starting live stream: {url}")

                    # Run stream in background task
                    stream_task = asyncio.create_task(
                        processor.process_stream(url, websocket, forced_language=lang))
                else:
                    url = data.get('url')
                    if url:
                        logger.warning("Using legacy message format")

                        if "youtube.com" not in url and "youtu.be" not in url:
                            await websocket.send_json({
                                "type": "error",
                                "data": "Only YouTube URLs supported"
                            })
                            continue

                        async with processor.lock:
                            processor.stop_flag = False
                            processor.connection_active = True   # 🔥 CRITICAL FIX
                            processor.transcript_buffer = []
                            processor.detected_language = None
 
                        stream_task = asyncio.create_task(
                            processor.process_stream(url, websocket)
                        )

            except asyncio.TimeoutError:
                logger.warning("WebSocket idle timeout")
                continue

            except WebSocketDisconnect:
                logger.info("Client disconnected")
                break

            except Exception as e:
                logger.error(f"WebSocket error: {e}", exc_info=True)
                try:
                    await websocket.send_json({
                        "type": "error",
                        "data": str(e)
                    })
                except Exception:
                    pass
                break

    finally:
        async with processor.lock:
            processor.stop_flag = True
            processor.connection_active = False
            processor.kill_processes()
            processor.active_websocket = None

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
    """Stop the live stream and generate final summaries"""
    try:
        processor = app.state.ws_processor
        
        async with processor.lock:
            processor.stop_flag = True
            processor.connection_active = False
            processor.kill_processes()
        
        all_text = " ".join([t.text for t in request.transcriptions if t.text.strip()])
        
        if not all_text.strip():
            return {
                "success": False, 
                "message": "No transcriptions available",
                "summaries": {lang: "No transcriptions available" for lang in ["en", "hi", "ml", "ta", "te"]}
            }
        
        summaries = await processor.generate_multilingual_summaries(
            all_text, 
            request.language or "en"
        )
        
        return {
            "success": True, 
            "message": "Stream stopped and summaries generated",
            "summaries": summaries
        }
        
    except Exception as e:
        logger.error(f"Error stopping stream: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error stopping stream: {str(e)}")


@app.post("/generate_summary/")
async def generate_summary(request: GenerateSummaryRequest):
    """Generate summaries from existing transcriptions"""
    try:
        processor = app.state.ws_processor
        
        all_text = " ".join([t.text for t in request.transcriptions if t.text.strip()])
        
        if not all_text.strip():
            raise HTTPException(
                status_code=400, 
                detail="No transcriptions available for summarization"
            )
        
        summaries = await processor.generate_multilingual_summaries(
            all_text, 
            request.language or "en"
        )
        
        return {
            "success": True, 
            "summaries": summaries
        }
        
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    signal.signal(signal.SIGINT, lambda s, f: shutdown_event.set())
    signal.signal(signal.SIGTERM, lambda s, f: shutdown_event.set())

    logger.info("🚀 Starting Multilingual Video Summarizer API")
    logger.info("📦 Features: Video Upload + Live Streaming + 5 Languages (en, hi, ml, ta, te)")
    logger.info("🌐 Server: http://localhost:8000")
    logger.info("📋 Docs: http://localhost:8000/docs")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        timeout_keep_alive=300,
        ws_ping_interval=20,
        ws_ping_timeout=20
    )
