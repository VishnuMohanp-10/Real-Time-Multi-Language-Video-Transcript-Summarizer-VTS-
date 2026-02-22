import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import './index.css';
import ProcessingStatus from './ProcessStatus';

const App = () => {
  const [loading, setLoading] = useState(true);
  const [backendError, setBackendError] = useState(null);
  const [activeTab, setActiveTab] = useState('upload');
  const [uploadProgress, setUploadProgress] = useState(0);
  const [videoResults, setVideoResults] = useState({
    transcript: '',
    language: '',
    translations: { english: '', hindi: '', malayalam: '', tamil: '', telugu: ''},
    summaries: { english: '', hindi: '', malayalam: '', tamil: '', telugu: '' },
    isLoading: false,
    processingState: 'idle',
    processingProgress: 0,
    processingMessage: ''
  });
  const [streamUrl, setStreamUrl] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [streamResults, setStreamResults] = useState({
    transcriptions: [],
    summaries: [],
    finalSummaries: { english: '', hindi: '', malayalam: '', tamil: '', telugu: '' }, // 🔥 FIXED: Added tamil & telugu
    language: null
  });
  const [error, setError] = useState(null);
  const wsRef = useRef(null);
  const cancelToken = useRef(null);
  const processingTimeout = useRef(null);
  const API_URL = "http://localhost:8000";

  const checkBackend = async () => {
    try {
      const response = await axios.get(`${API_URL}/`, { timeout: 3000 });
      return response.data.status === 'running';
    } catch (err) {
      setBackendError('Backend server is not running. Please start the Python server first.');
      setLoading(false);
      console.error('Backend connection error:', err);
      return false;
    }
  };

  useEffect(() => {
    checkBackend().then(isRunning => {
      if (isRunning) {
        setLoading(false);
      }
    });

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (cancelToken.current) {
        cancelToken.current.cancel();
      }
      if (processingTimeout.current) {
        clearTimeout(processingTimeout.current);
      }
    };
  }, []);

  // 🔥 FIXED: Removed duplicate useEffect, consolidated WebSocket handling
  const handleVideoUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    try {
      setError(null);
      setUploadProgress(0);
      setVideoResults(prev => ({ 
        ...prev, 
        isLoading: true,
        processingState: 'uploading',
        processingProgress: 0,
        processingMessage: 'Starting upload...'
      }));

      const formData = new FormData();
      formData.append('file', file);
      
      cancelToken.current = axios.CancelToken.source();
      processingTimeout.current = setTimeout(() => {
        cancelToken.current.cancel('Processing timeout');
      }, 600000);

      const response = await axios.post(`${API_URL}/upload/`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        cancelToken: cancelToken.current.token,
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round(
            (progressEvent.loaded * 100) / progressEvent.total
          );
          setUploadProgress(percentCompleted);
          setVideoResults(prev => ({
            ...prev,
            processingProgress: Math.min(percentCompleted, 30),
            processingMessage: `Uploading video (${percentCompleted}%)`
          }));
        }
      });
       if (!response.data || !response.data.success) {
        throw new Error(response.data?.error || 'Unknown processing error or empty response from server.');
      }

      clearTimeout(processingTimeout.current);

      setVideoResults({
        transcript: response.data.transcript || '',
        language: response.data.language || 'en',
        translations: {
          english: response.data.translations?.en || 'Unavailable',
          hindi: response.data.translations?.hi || 'Unavailable',
          malayalam: response.data.translations?.ml || 'Unavailable',
          tamil: response.data.translations?.ta || 'Unavailable',
          telugu: response.data.translations?.te || 'Unavailable'
        },
        summaries: {
          english: response.data.summaries?.en || 'Unavailable',
          hindi: response.data.summaries?.hi || 'Unavailable',
          malayalam: response.data.summaries?.ml || 'Unavailable',
          tamil: response.data.summaries?.ta || 'Unavailable',
          telugu: response.data.summaries?.te || 'Unavailable'
        },
        isLoading: false,
        processingState: 'complete',
        processingProgress: 100,
        processingMessage: 'Processing complete!'
      });
      
    } catch (err) {
      clearTimeout(processingTimeout.current);
      let errorMessage = 'Processing failed';
      if (axios.isCancel(err)) {
        errorMessage = err.message || 'Processing cancelled';
      } else if (err.response) {
        errorMessage = err.response.data?.detail || `Server error (${err.response.status})`;
      } else if (err.request) {
        errorMessage = 'No response from server';
      } else {
        errorMessage = err.message || 'Unknown error';
      }
      
      setError(errorMessage);
      setVideoResults(prev => ({ 
        ...prev, 
        isLoading: false, 
        processingState: 'error',
        processingMessage: errorMessage
      }));
    }
  };

  const handleVideoCancel = () => {
    if (cancelToken.current) {
      cancelToken.current.cancel('Processing cancelled by user');
    }
    setVideoResults(prev => ({ 
      ...prev, 
      isLoading: false, 
      processingState: 'cancelled',
      processingMessage: 'Processing cancelled by user'
    }));
    setUploadProgress(0);
  };

  const handleStreamSubmit = async (e) => {
    e.preventDefault();
    
    if (!await checkBackend()) {
      setError("Backend server is not available");
      return;
    }
  
    try {
      // Clear previous results
      setStreamResults({
        transcriptions: [],
        summaries: [],
        finalSummaries: { english: '', hindi: '', malayalam: '', tamil: '', telugu: '' },
        language: null
      });
      setError(null);
  
      const wsUrl = `ws://localhost:8000/ws/live`;
      const ws = new WebSocket(wsUrl);
  
      ws.onopen = () => {
        console.log("✅ WebSocket connected");
        
        // 🔥 CRITICAL FIX: Send correct message format
        const message = {
          url: streamUrl  // Backend expects this for legacy format
        };
        
        console.log("📤 Sending message:", message);
        ws.send(JSON.stringify(message));
        
        setIsStreaming(true);
        setConnectionStatus('connecting');
      };

      // 🔥 FIXED: Proper message handling
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          console.log("📥 Received message:", data.type, data);
          
          if (data.type === 'status') {
            console.log("📊 Status:", data.data);
            setConnectionStatus(data.data.includes('Connecting') || data.data.includes('connecting') ? 'connecting' : 'processing');
          }
          else if (data.type === 'language_detected') {
            console.log("🌍 Language detected:", data.data);
            setStreamResults(prev => ({
              ...prev,
              language: data.data
            }));
          }
          else if (data.type === 'transcription') {
            console.log("📝 Transcription received:", data.data);
            setStreamResults(prev => ({
              ...prev,
              transcriptions: [...prev.transcriptions.slice(-9), {
                text: data.data,
                language: data.language,
                timestamp: new Date(data.timestamp).toLocaleTimeString(),
                isNew: true
              }],
              language: data.language || prev.language
            }));
            
            // Remove "new" flag after animation
            setTimeout(() => {
              setStreamResults(prev => ({
                ...prev,
                transcriptions: prev.transcriptions.map(item => ({
                  ...item,
                  isNew: false
                }))
              }));
            }, 1500);
          } 
          else if (data.type === 'summary') {
            console.log("📋 Summary received:", data.data);
            setStreamResults(prev => ({
              ...prev,
              summaries: [...prev.summaries.slice(-4), {
                english: data.data.en || 'Translation unavailable',
                hindi: data.data.hi || 'Translation unavailable',
                malayalam: data.data.ml || 'Translation unavailable',
                tamil: data.data.ta || 'Translation unavailable',
                telugu: data.data.te || 'Translation unavailable',
                timestamp: new Date(data.timestamp).toLocaleTimeString(),
                isNew: true
              }]
            }));
            
            setTimeout(() => {
              setStreamResults(prev => ({
                ...prev,
                summaries: prev.summaries.map(item => ({
                  ...item,
                  isNew: false
                }))
              }));
            }, 1500);
          }
          else if (data.type === 'final_summary') {
            console.log("🏁 Final summary received:", data.data);
            setStreamResults(prev => ({
              ...prev,
              finalSummaries: {
                english: data.data.en || 'No summary available',
                hindi: data.data.hi || 'No summary available',
                malayalam: data.data.ml || 'No summary available',
                tamil: data.data.ta || 'No summary available',
                telugu: data.data.te || 'No summary available'
              }
            }));
          }
          else if (data.type === 'error') {
            console.error("❌ Error from backend:", data.data);
            setError(data.data);
            handleStopStream();
          }
          else if (data.type === 'stopped') {
            console.log("⏹️ Stream stopped");
            setIsStreaming(false);
            setConnectionStatus('disconnected');
          }
          else {
            console.log("❓ Unknown message type:", data.type);
          }
        } catch (err) {
          console.error('❌ Error processing message:', err, event.data);
        }
      };
  
      ws.onerror = (error) => {
        console.error("❌ WebSocket error:", error);
        setError("WebSocket connection failed");
        setIsStreaming(false);
        setConnectionStatus('disconnected');
      };
  
      ws.onclose = (event) => {
        console.log("🔌 WebSocket closed:", event.code, event.reason);
        setIsStreaming(false);
        setConnectionStatus('disconnected');
      };
  
      wsRef.current = ws;
      
    } catch (err) {
      console.error("❌ Connection error:", err);
      setError(`Connection failed: ${err.message}`);
      setIsStreaming(false);
      setConnectionStatus('disconnected');
    }
  };
  
  const handleStopStream = async () => {
    try {
      console.log("🛑 Stopping stream...");
      
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        // 🔥 FIXED: Send proper stop command
        wsRef.current.send(JSON.stringify({ action: 'stop' }));
        console.log("📤 Sent stop command");
        
        // Wait for final summary (up to 5 seconds)
        await new Promise(resolve => setTimeout(resolve, 5000));
      }

      // Close WebSocket
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      
    } catch (err) {
      console.error('❌ Stop error:', err);
    } finally {
      setIsStreaming(false);
      setConnectionStatus('disconnected');
    }
  };

  const getLanguageName = (code) => {
    const names = {
      'en': 'English',
      'hi': 'Hindi',
      'ml': 'Malayalam',
      'ta': 'Tamil',
      'te': 'Telugu'
    };
    return names[code] || code;
  };

  if (loading) {
    return (
      <div className="loading-screen">
        <div className="spinner"></div>
        <p>Connecting to backend server...</p>
        {backendError && (
          <div className="error-message">
            <p>{backendError}</p>
            <div className="solution-steps">
              <h3>To fix this:</h3>
              <ol>
                <li>Open a terminal in your backend directory</li>
                <li>Run: <code>python main.py</code></li>
                <li>Make sure the server starts on port 8000</li>
                <li>Refresh this page after the backend is running</li>
              </ol>
            </div>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="app-container">
      <header className="app-header">
        <h1 className="gradient-title">Multilingual Video Summarizer</h1>
        <p className="subtitle">Upload videos or process live streams in English, Hindi, Malayalam, Tamil, and Telugu</p>
      </header>

      <div className="tab-container">
        <button 
          className={`tab-button ${activeTab === 'upload' ? 'active' : ''}`}
          onClick={() => setActiveTab('upload')}
        >
          Video Upload
        </button>
        <button 
          className={`tab-button ${activeTab === 'stream' ? 'active' : ''}`}
          onClick={() => setActiveTab('stream')}
        >
          Live Stream
        </button>
      </div>

      {activeTab === 'upload' ? (
        <>
          <section className="upload-section">
            <h2>Video Processing</h2>
            <div className="upload-card">
              <label className="file-upload-button">
                <input 
                  type="file" 
                  accept="video/*" 
                  onChange={handleVideoUpload} 
                  style={{ display: 'none' }} 
                  disabled={videoResults.isLoading}
                />
                <span className="upload-icon">📁</span>
                <span>Select Video File</span>
              </label>
              
              {uploadProgress > 0 && (
                <div className="progress-container">
                  <div className="progress-indicator">
                    <div
                      className="progress-bar"
                      style={{ width: `${uploadProgress}%` }}
                    ></div>
                  </div>
                  <span className="progress-text">{uploadProgress}%</span>
                </div>
              )}
              
              {(videoResults.isLoading || videoResults.processingState !== 'idle') && (
                <div className="processing-status">
                  <div className="processing-steps">
                    <div className={`processing-step ${['uploading', 'extracting', 'transcribing', 'translating', 'summarizing', 'complete'].includes(videoResults.processingState) ? 'completed' : ''}`}>
                      <div className="processing-step-indicator">1</div>
                      <span>Uploading</span>
                    </div>
                    <div className={`processing-step ${['extracting', 'transcribing', 'translating', 'summarizing', 'complete'].includes(videoResults.processingState) ? 'completed' : ''}`}>
                      <div className="processing-step-indicator">2</div>
                      <span>Extracting</span>
                    </div>
                    <div className={`processing-step ${['transcribing', 'translating', 'summarizing', 'complete'].includes(videoResults.processingState) ? 'completed' : ''}`}>
                      <div className="processing-step-indicator">3</div>
                      <span>Transcribing</span>
                    </div>
                    <div className={`processing-step ${['translating', 'summarizing', 'complete'].includes(videoResults.processingState) ? 'completed' : ''}`}>
                      <div className="processing-step-indicator">4</div>
                      <span>Translating</span>
                    </div>
                    <div className={`processing-step ${['summarizing', 'complete'].includes(videoResults.processingState) ? 'completed' : ''}`}>
                      <div className="processing-step-indicator">5</div>
                      <span>Summarizing</span>
                    </div>
                  </div>
                  
                  <div className="processing-progress">
                    <div 
                      className="processing-progress-bar" 
                      style={{ width: `${videoResults.processingProgress}%` }}
                    ></div>
                  </div>
                  
                  <div className="processing-message">
                    <p>{videoResults.processingMessage}</p>
                    {error && <p className="error-text">{error}</p>}
                    <p className="processing-time-note">
                      Processing may take several minutes depending on video length.
                    </p>
                  </div>
                </div>
              )}
              
              {uploadProgress > 0 && uploadProgress < 100 && (
                <button
                  className="cancel-btn"
                  onClick={handleVideoCancel}
                  disabled={!videoResults.isLoading}
                >
                  Cancel Processing
                </button>
              )}
            </div>
          </section>

          {videoResults.transcript && (
            <div className="results-section">
              <div className="result-header">
                <h2>Video Analysis Results</h2>
                {videoResults.language && (
                  <span className="language-tag">
                    {getLanguageName(videoResults.language)}
                  </span>
                )}
              </div>

              <div className="results-grid">
                <div className="result-card">
                  <h3>Original Transcript ({getLanguageName(videoResults.language)})</h3>
                  <div className="scrollable-content">
                    <p>{videoResults.transcript}</p>
                  </div>
              </div>

                {Object.entries(videoResults.translations).map(([lang, text]) => (
                  <div key={lang} className="result-card">
                    <h3>{getLanguageName(lang)} Translation</h3>
                    <div className="scrollable-content">
                      <p>{text}</p>
                      {text === 'Unavailable' && (
                        <p className="translation-unavailable">Translation not available</p>
                      )}
                    </div>
                  </div>
                ))}

                {Object.entries(videoResults.summaries).map(([lang, text]) => (
                  <div key={lang} className="result-card">
                    <h3>{getLanguageName(lang)} Summary</h3>
                    <div className="scrollable-content">
                      <p>{text}</p>
                      {text === 'Unavailable' && (
                        <p className="translation-unavailable">Summary not available</p>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      ) : (
        <>
          <section className="stream-section">
            <h2>Live Stream Processing</h2>
            <form className="stream-form" onSubmit={handleStreamSubmit}>
              <input
                type="text"
                className="stream-input"
                placeholder="Enter YouTube live stream URL"
                value={streamUrl}
                onChange={(e) => setStreamUrl(e.target.value)}
                disabled={isStreaming}
              />
              {isStreaming ? (
                <button
                  type="button"
                  className="stream-button stop"
                  onClick={handleStopStream}
                >
                  Stop Processing
                </button>
              ) : (
                <button
                  type="submit"
                  className="stream-button"
                  disabled={!streamUrl}
                >
                  Start Processing
                </button>
              )}
            </form>
            
            {error && <div className="error-message">{error}</div>}
            
            <ProcessingStatus 
              isStreaming={isStreaming} 
              isUploading={false} 
              status={connectionStatus}
            />
          </section>

          {(isStreaming || streamResults.transcriptions.length > 0 || streamResults.summaries.length > 0 || 
            Object.values(streamResults.finalSummaries).some(s => s)) && (
            <div className="results-section">
              <div className="result-header">
                <h2>Live Stream Results</h2>
                {streamResults.language && (
                  <span className="language-tag">
                    {getLanguageName(streamResults.language)}
                  </span>
                )}
                {isStreaming && (
                  <div className="processing-indicator">
                    <div className="pulse-dot"></div>
                    <span>Processing live stream...</span>
                  </div>
                )}
              </div>

              <div className="live-results-grid">
                <div className="live-transcript-container">
                  <h3>Live Transcription History</h3>
                  <div className="live-transcripts">
                    {streamResults.transcriptions.length > 0 ? (
                      streamResults.transcriptions.map((item, index) => (
                        <div key={index} className={`transcript-item ${item.isNew ? 'new-item' : ''}`}>
                          <div className="meta">
                            <span className="timestamp">{item.timestamp}</span>
                            {item.language && (
                              <span className="language-badge">
                                {getLanguageName(item.language)}
                              </span>
                            )}
                          </div>
                          <p>{item.text}</p>
                        </div>
                      ))
                    ) : (
                      <p className="empty-state">Waiting for transcriptions...</p>
                    )}
                  </div>
                </div>

                <div className="live-summary-container">
                  <div className="final-summary-box">
                    <h3>Final Summaries (All 5 Languages)</h3>
                    <div className="final-summary-content">
                      {Object.entries(streamResults.finalSummaries).map(([lang, text]) => (
                        <div key={lang}>
                          <h4>{getLanguageName(lang)}</h4>
                          <p>{text || "Will appear when stream is stopped"}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                  
                  <div className="summary-history">
                    <h4>Summary History</h4>
                    <div className="summary-items">
                      {streamResults.summaries.length > 0 ? (
                        streamResults.summaries.map((item, index) => (
                          <div key={index} className={`summary-item ${item.isNew ? 'new-item' : ''}`}>
                            <div className="meta">
                              <span className="timestamp">{item.timestamp}</span>
                            </div>
                            {Object.entries(item).filter(([key]) => key !== 'timestamp' && key !== 'isNew').map(([lang, text]) => (
                              <div key={lang}>
                                <h5>{getLanguageName(lang)}</h5>
                                <p>{text}</p>
                              </div>
                            ))}
                          </div>
                        ))
                      ) : (
                        <p className="empty-state">Summaries will appear every few seconds</p>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default App;
