import React, { useState, useRef, useEffect, useCallback } from 'react';
import axios from 'axios';
import './index.css';
import ProcessingStatus from './ProcessStatus';

const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

// ─── language helpers ──────────────────────────────────────────────────────

const LANGUAGES = [
  { key: 'english',  label: 'English',   flag: '🇬🇧', code: 'en' },
  { key: 'hindi',    label: 'Hindi',     flag: '🇮🇳', code: 'hi' },
  { key: 'malayalam',label: 'Malayalam', flag: '🇮🇳', code: 'ml' },
  { key: 'tamil',    label: 'Tamil',     flag: '🇮🇳', code: 'ta' },
  { key: 'telugu',   label: 'Telugu',    flag: '🇮🇳', code: 'te' },
];

function getLanguageFlag(lang) {
  const flags = { english:'🇬🇧', hindi:'🇮🇳', malayalam:'🇮🇳', tamil:'🇮🇳', telugu:'🇮🇳', en:'🇬🇧', hi:'🇮🇳', ml:'🇮🇳', ta:'🇮🇳', te:'🇮🇳' };
  return flags[lang] || '🌐';
}

function getLanguageName(code) {
  const names = { en:'English', english:'English', hi:'Hindi', hindi:'Hindi', ml:'Malayalam', malayalam:'Malayalam', ta:'Tamil', tamil:'Tamil', te:'Telugu', telugu:'Telugu' };
  return names[code] || code;
}


// ============================================================================
//  LOGIN / REGISTER SCREEN
// ============================================================================

function AuthScreen({ onLogin }) {
  const [mode,     setMode]     = useState('login');   // 'login' | 'register'
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [showPw,   setShowPw]   = useState(false);
  const [loading,  setLoading]  = useState(false);
  const [error,    setError]    = useState('');
  const [success,  setSuccess]  = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setSuccess('');
    if (!username.trim() || !password.trim()) {
      setError('Please fill in all fields');
      return;
    }
    setLoading(true);
    try {
      const endpoint = mode === 'login' ? '/auth/login' : '/auth/register';
      const res = await axios.post(`${API_URL}${endpoint}`, { username: username.trim(), password: password.trim() });
      if (res.data.success) {
        setSuccess(mode === 'login' ? 'Login successful! Redirecting…' : 'Account created! Redirecting…');
        setTimeout(() => onLogin(res.data.token, res.data.username), 700);
      }
    } catch (err) {
      const msg = err.response?.data?.detail || 'Something went wrong. Please try again.';
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const switchMode = () => {
    setMode(m => m === 'login' ? 'register' : 'login');
    setError('');
    setSuccess('');
    setUsername('');
    setPassword('');
  };

  return (
    <div className="auth-screen">
      {/* Ambient orbs */}
      <div className="auth-orb auth-orb-1" />
      <div className="auth-orb auth-orb-2" />
      <div className="auth-orb auth-orb-3" />

      <div className="auth-card">
        {/* Logo / branding */}
        <div className="auth-logo">
          <div className="auth-logo-icon">
            <span>VTS</span>
          </div>
          <h1 className="auth-app-name">Real-Time Multilingual Video Summarizer</h1>
          <p className="auth-app-tagline">Real-time transcription in 5 languages</p>
        </div>

        {/* Language pill strip */}
        <div className="auth-lang-strip">
          {LANGUAGES.map(l => (
            <span key={l.key} className="auth-lang-pill">{l.flag} {l.label}</span>
          ))}
        </div>

        {/* Mode toggle */}
        <div className="auth-mode-toggle">
          <button className={`auth-mode-btn ${mode === 'login'    ? 'active' : ''}`} onClick={() => { setMode('login');    setError(''); setSuccess(''); }}>Sign In</button>
          <button className={`auth-mode-btn ${mode === 'register' ? 'active' : ''}`} onClick={() => { setMode('register'); setError(''); setSuccess(''); }}>Register</button>
        </div>

        {/* Form */}
        <form className="auth-form" onSubmit={handleSubmit}>
          <div className="auth-field">
            <label className="auth-label">Username</label>
            <div className="auth-input-wrap">
              <span className="auth-input-icon">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>
              </span>
              <input
                className="auth-input"
                type="text"
                placeholder={mode === 'register' ? 'Choose a username (min 3 chars)' : 'Enter your username'}
                value={username}
                onChange={e => setUsername(e.target.value)}
                disabled={loading}
                autoComplete="username"
                autoFocus
              />
            </div>
          </div>

          <div className="auth-field">
            <label className="auth-label">Password</label>
            <div className="auth-input-wrap">
              <span className="auth-input-icon">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>
              </span>
              <input
                className="auth-input"
                type={showPw ? 'text' : 'password'}
                placeholder={mode === 'register' ? 'Create a password (min 6 chars)' : 'Enter your password'}
                value={password}
                onChange={e => setPassword(e.target.value)}
                disabled={loading}
                autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
              />
              <button type="button" className="auth-pw-toggle" onClick={() => setShowPw(v => !v)} tabIndex={-1}>
                {showPw
                  ? <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94"/><path d="M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19"/><line x1="1" y1="1" x2="23" y2="23"/></svg>
                  : <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>
                }
              </button>
            </div>
          </div>

          {error   && <div className="auth-error">{error}</div>}
          {success && <div className="auth-success">{success}</div>}

          <button className="auth-submit-btn" type="submit" disabled={loading}>
            {loading
              ? <span className="auth-spinner" />
              : mode === 'login' ? 'Sign In →' : 'Create Account →'
            }
          </button>
        </form>

        <p className="auth-switch-text">
          {mode === 'login' ? "Don't have an account? " : 'Already have an account? '}
          <button className="auth-switch-link" onClick={switchMode}>
            {mode === 'login' ? 'Register here' : 'Sign in here'}
          </button>
        </p>
      </div>
    </div>
  );
}


// ============================================================================
//  DOWNLOAD BUTTON  (reusable)
// ============================================================================

function DownloadButton({ content, language, contentType, token, className = '' }) {
  const [state, setState] = useState('idle'); // idle | loading | done | error

  const handleDownload = async () => {
    if (!content || state === 'loading') return;
    setState('loading');
    try {
      const res = await axios.post(
        `${API_URL}/download/text`,
        { content, language, content_type: contentType, token },
        { responseType: 'blob' }
      );
      const url      = window.URL.createObjectURL(new Blob([res.data]));
      const link     = document.createElement('a');
      const langName = getLanguageName(language);
      link.href      = url;
      link.download  = `VTS_${contentType}_${langName}_${Date.now()}.txt`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
      setState('done');
      setTimeout(() => setState('idle'), 2000);
    } catch (err) {
      console.error('Download failed:', err);
      setState('error');
      setTimeout(() => setState('idle'), 2000);
    }
  };

  const icon = state === 'loading' ? '⏳'
             : state === 'done'    ? '✅'
             : state === 'error'   ? '❌'
             : '⬇';

  return (
    <button
      className={`download-btn ${state !== 'idle' ? `download-btn--${state}` : ''} ${className}`}
      onClick={handleDownload}
      disabled={!content || state === 'loading'}
      title={`Download ${contentType} (${getLanguageName(language)})`}
    >
      <span className="download-btn-icon">{icon}</span>
      <span className="download-btn-label">
        {state === 'loading' ? 'Saving…' : state === 'done' ? 'Saved!' : state === 'error' ? 'Failed' : 'Download'}
      </span>
    </button>
  );
}


// ============================================================================
//  MAIN APP
// ============================================================================

const App = () => {
  // ── auth state ────────────────────────────────────────────────────────────
  const [authToken,  setAuthToken]  = useState(() => sessionStorage.getItem('vts_token')    || null);
  const [authUser,   setAuthUser]   = useState(() => sessionStorage.getItem('vts_username') || null);
  const [isLoggedIn, setIsLoggedIn] = useState(!!sessionStorage.getItem('vts_token'));

  // ── app state ─────────────────────────────────────────────────────────────
  const [loading,      setLoading]      = useState(true);
  const [backendError, setBackendError] = useState(null);
  const [activeTab,    setActiveTab]    = useState('upload');
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error,        setError]        = useState(null);

  const [videoResults, setVideoResults] = useState({
    transcript: '', language: '',
    translations: { english:'', hindi:'', malayalam:'', tamil:'', telugu:'' },
    summaries:    { english:'', hindi:'', malayalam:'', tamil:'', telugu:'' },
    isLoading: false, processingState:'idle', processingProgress:0, processingMessage:''
  });

  const [streamUrl,        setStreamUrl]        = useState('');
  const [isStreaming,      setIsStreaming]       = useState(false);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [selectedLang,     setSelectedLang]     = useState('english');
  const [dropdownOpen,     setDropdownOpen]     = useState(false);

  const [streamResults, setStreamResults] = useState({
    fullTranscript: '',
    liveTranslations: { english:'', hindi:'', malayalam:'', tamil:'', telugu:'' },
    recentFragment: '',
    language: null
  });

  const wsRef              = useRef(null);
  const cancelToken        = useRef(null);
  const processingTimeout  = useRef(null);
  const transcriptEndRef   = useRef(null);
  const dropdownRef        = useRef(null);

  // ── auth handlers ─────────────────────────────────────────────────────────

  const handleLogin = useCallback((token, username) => {
    sessionStorage.setItem('vts_token',    token);
    sessionStorage.setItem('vts_username', username);
    setAuthToken(token);
    setAuthUser(username);
    setIsLoggedIn(true);
  }, []);

  const handleLogout = useCallback(async () => {
    try {
      if (authToken) await axios.post(`${API_URL}/auth/logout?token=${authToken}`);
    } catch (_) {}
    sessionStorage.removeItem('vts_token');
    sessionStorage.removeItem('vts_username');
    setAuthToken(null);
    setAuthUser(null);
    setIsLoggedIn(false);
    if (wsRef.current) wsRef.current.close();
  }, [authToken]);

  // ── backend check ─────────────────────────────────────────────────────────

  const checkBackend = async () => {
    try {
      const res = await axios.get(`${API_URL}/`, { timeout: 60000 });
      return res.data.status === 'running';
    } catch (err) {
      setBackendError('Backend server is not running. Please start the Python server first.');
      setLoading(false);
      return false;
    }
  };

  useEffect(() => {
    checkBackend().then(ok => { if (ok) setLoading(false); });
    return () => {
      if (wsRef.current)           wsRef.current.close();
      if (cancelToken.current)     cancelToken.current.cancel();
      if (processingTimeout.current) clearTimeout(processingTimeout.current);
    };
  }, []);

  // close dropdown on outside click
  useEffect(() => {
    const handler = (e) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target)) setDropdownOpen(false);
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  // auto-scroll transcript
  useEffect(() => {
    if (transcriptEndRef.current) transcriptEndRef.current.scrollIntoView({ behavior: 'smooth' });
  }, [streamResults.fullTranscript]);

  // ── video upload ──────────────────────────────────────────────────────────

  const handleVideoUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    try {
      setError(null);
      setUploadProgress(0);
      setVideoResults(prev => ({ ...prev, isLoading:true, processingState:'uploading', processingProgress:0, processingMessage:'Starting upload…' }));
      const formData = new FormData();
      formData.append('file', file);
      cancelToken.current    = axios.CancelToken.source();
      processingTimeout.current = setTimeout(() => cancelToken.current.cancel('Processing timeout'), 600000);
      const response = await axios.post(`${API_URL}/upload/`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        cancelToken: cancelToken.current.token,
        onUploadProgress: (evt) => {
          const pct = Math.round((evt.loaded * 100) / evt.total);
          setUploadProgress(pct);
          setVideoResults(prev => ({ ...prev, processingProgress: Math.min(pct, 30), processingMessage:`Uploading video (${pct}%)` }));
        }
      });
      if (!response.data?.success) throw new Error(response.data?.error || 'Unknown processing error');
      clearTimeout(processingTimeout.current);
      setVideoResults({
        transcript: response.data.transcript || '',
        language:   response.data.language   || 'en',
        translations: {
          english:   response.data.translations?.en || 'Unavailable',
          hindi:     response.data.translations?.hi || 'Unavailable',
          malayalam: response.data.translations?.ml || 'Unavailable',
          tamil:     response.data.translations?.ta || 'Unavailable',
          telugu:    response.data.translations?.te || 'Unavailable',
        },
        summaries: {
          english:   response.data.summaries?.en || 'Unavailable',
          hindi:     response.data.summaries?.hi || 'Unavailable',
          malayalam: response.data.summaries?.ml || 'Unavailable',
          tamil:     response.data.summaries?.ta || 'Unavailable',
          telugu:    response.data.summaries?.te || 'Unavailable',
        },
        isLoading:false, processingState:'complete', processingProgress:100, processingMessage:'Processing complete!'
      });
    } catch (err) {
      clearTimeout(processingTimeout.current);
      const msg = axios.isCancel(err) ? (err.message || 'Cancelled')
                : err.response ? (err.response.data?.detail || `Server error (${err.response.status})`)
                : err.request  ? 'No response from server'
                : err.message  || 'Unknown error';
      setError(msg);
      setVideoResults(prev => ({ ...prev, isLoading:false, processingState:'error', processingMessage:msg }));
    }
  };

  const handleVideoCancel = () => {
    if (cancelToken.current) cancelToken.current.cancel('Cancelled by user');
    setVideoResults(prev => ({ ...prev, isLoading:false, processingState:'cancelled', processingMessage:'Cancelled by user' }));
    setUploadProgress(0);
  };

  // ── live stream ───────────────────────────────────────────────────────────

  const handleStreamSubmit = async (e) => {
    e.preventDefault();
    if (!await checkBackend()) { setError("Backend server is not available"); return; }
    try {
      setStreamResults({ fullTranscript:'', liveTranslations:{ english:'', hindi:'', malayalam:'', tamil:'', telugu:'' }, recentFragment:'', language:null });
      setError(null);
      const wsUrl = `${API_URL.replace('http', 'ws')}/ws/live`;
      const ws    = new WebSocket(wsUrl);
      ws.onopen = () => {
        ws.send(JSON.stringify({ url: streamUrl }));
        setIsStreaming(true);
        setConnectionStatus('connecting');
      };
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'status') {
            setConnectionStatus(data.data.includes('Connecting') || data.data.includes('connecting') ? 'connecting' : 'processing');
          } else if (data.type === 'language_detected') {
            setStreamResults(prev => ({ ...prev, language: data.data }));
          } else if (data.type === 'transcription') {
            setStreamResults(prev => ({
              ...prev,
              fullTranscript: prev.fullTranscript ? prev.fullTranscript + ' ' + data.data : data.data,
              recentFragment: data.data,
              language: data.language || prev.language
            }));
            setTimeout(() => setStreamResults(prev => ({ ...prev, recentFragment:'' })), 2000);
          } else if (data.type === 'summary') {
            setStreamResults(prev => ({
              ...prev,
              liveTranslations: {
                english:   (prev.liveTranslations.english   ? prev.liveTranslations.english   + ' ' : '') + (data.data.en || ''),
                hindi:     (prev.liveTranslations.hindi     ? prev.liveTranslations.hindi     + ' ' : '') + (data.data.hi || ''),
                malayalam: (prev.liveTranslations.malayalam ? prev.liveTranslations.malayalam + ' ' : '') + (data.data.ml || ''),
                tamil:     (prev.liveTranslations.tamil     ? prev.liveTranslations.tamil     + ' ' : '') + (data.data.ta || ''),
                telugu:    (prev.liveTranslations.telugu    ? prev.liveTranslations.telugu    + ' ' : '') + (data.data.te || ''),
              }
            }));
          } else if (data.type === 'error') {
            setError(data.data);
            handleStopStream();
          } else if (data.type === 'stopped') {
            setIsStreaming(false);
            setConnectionStatus('disconnected');
          }
        } catch (err) { console.error('Message parse error:', err); }
      };
      ws.onerror = () => { setError("WebSocket connection failed"); setIsStreaming(false); setConnectionStatus('disconnected'); };
      ws.onclose = () => { setIsStreaming(false); setConnectionStatus('disconnected'); };
      wsRef.current = ws;
    } catch (err) {
      setError(`Connection failed: ${err.message}`);
      setIsStreaming(false);
      setConnectionStatus('disconnected');
    }
  };

  const handleStopStream = async () => {
    try {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ action: 'stop' }));
        await new Promise(r => setTimeout(r, 3000));
      }
      if (wsRef.current) { wsRef.current.close(); wsRef.current = null; }
    } catch (err) { console.error('Stop error:', err); }
    finally { setIsStreaming(false); setConnectionStatus('disconnected'); }
  };

  // ── derived values ────────────────────────────────────────────────────────

  const activeLang    = LANGUAGES.find(l => l.key === selectedLang) || LANGUAGES[0];
  const activeText    = streamResults.liveTranslations[selectedLang];

  // ── loading screen ────────────────────────────────────────────────────────

  if (loading) {
    return (
      <div className="loading-screen">
        <div className="spinner" />
        <p>Connecting to backend server…</p>
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

  // ── auth gate ─────────────────────────────────────────────────────────────

  if (!isLoggedIn) {
    return <AuthScreen onLogin={handleLogin} />;
  }

  // ── main app ──────────────────────────────────────────────────────────────

  return (
    <div className="app-container">
      {/* ── Header ── */}
      <header className="app-header">
        <h1 className="gradient-title">Real-Time Multilingual Video Summarizer</h1>
        <p className="subtitle">Upload videos or process live streams in English, Hindi, Malayalam, Tamil, and Telugu</p>

        {/* User badge + logout */}
        <div className="user-badge">
          <div className="user-badge-avatar">{authUser?.[0]?.toUpperCase()}</div>
          <span className="user-badge-name">{authUser}</span>
          <button className="logout-btn" onClick={handleLogout} title="Sign out">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/>
              <polyline points="16 17 21 12 16 7"/>
              <line x1="21" y1="12" x2="9" y2="12"/>
            </svg>
            <span>Sign out</span>
          </button>
        </div>
      </header>

      {/* ── Tabs ── */}
      <div className="tab-container">
        <button className={`tab-button ${activeTab === 'upload' ? 'active' : ''}`} onClick={() => setActiveTab('upload')}>Video Upload</button>
        <button className={`tab-button ${activeTab === 'stream' ? 'active' : ''}`} onClick={() => setActiveTab('stream')}>Live Stream</button>
      </div>

      {/* ══════════════════════════════ UPLOAD TAB ══════════════════════════ */}
      {activeTab === 'upload' ? (
        <>
          <section className="upload-section">
            <h2>Video Processing</h2>
            <div className="upload-card">
              <label className="file-upload-button">
                <input type="file" accept="video/*" onChange={handleVideoUpload} style={{ display:'none' }} disabled={videoResults.isLoading} />
                <span className="upload-icon">📁</span>
                <span>Select Video File</span>
              </label>

              {uploadProgress > 0 && (
                <div className="progress-container">
                  <div className="progress-indicator">
                    <div className="progress-bar" style={{ width:`${uploadProgress}%` }} />
                  </div>
                  <span className="progress-text">{uploadProgress}%</span>
                </div>
              )}

              {(videoResults.isLoading || videoResults.processingState !== 'idle') && (
                <div className="processing-status">
                  <div className="processing-steps">
                    {['Uploading','Extracting','Transcribing','Translating','Summarizing'].map((label, i) => {
                      const states = [
                        ['uploading','extracting','transcribing','translating','summarizing','complete'],
                        ['extracting','transcribing','translating','summarizing','complete'],
                        ['transcribing','translating','summarizing','complete'],
                        ['translating','summarizing','complete'],
                        ['summarizing','complete'],
                      ];
                      return (
                        <div key={label} className={`processing-step ${states[i].includes(videoResults.processingState) ? 'completed' : ''}`}>
                          <div className="processing-step-indicator">{i + 1}</div>
                          <span>{label}</span>
                        </div>
                      );
                    })}
                  </div>
                  <div className="processing-progress">
                    <div className="processing-progress-bar" style={{ width:`${videoResults.processingProgress}%` }} />
                  </div>
                  <div className="processing-message">
                    <p>{videoResults.processingMessage}</p>
                    {error && <p className="error-text">{error}</p>}
                    <p className="processing-time-note">Processing may take several minutes depending on video length.</p>
                  </div>
                </div>
              )}

              {uploadProgress > 0 && uploadProgress < 100 && (
                <button className="cancel-btn" onClick={handleVideoCancel} disabled={!videoResults.isLoading}>Cancel Processing</button>
              )}
            </div>
          </section>

          {/* ── Upload results ── */}
          {videoResults.transcript && (
            <div className="results-section">
              <div className="result-header">
                <h2>Video Analysis Results</h2>
                {videoResults.language && <span className="language-tag">{getLanguageName(videoResults.language)}</span>}
              </div>

              <div className="results-grid">
                {/* Original transcript */}
                <div className="result-card">
                  <div className="result-card-header">
                    <h3>Original Transcript ({getLanguageName(videoResults.language)})</h3>
                    <DownloadButton content={videoResults.transcript} language={videoResults.language} contentType="transcript" token={authToken} />
                  </div>
                  <div className="scrollable-content"><p>{videoResults.transcript}</p></div>
                </div>

                {/* Translations */}
                {Object.entries(videoResults.translations).map(([lang, text]) => (
                  <div key={lang} className="result-card">
                    <div className="result-card-header">
                      <h3>{getLanguageFlag(lang)} {getLanguageName(lang)} Translation</h3>
                      <DownloadButton content={text !== 'Unavailable' ? text : ''} language={lang} contentType="translation" token={authToken} />
                    </div>
                    <div className="scrollable-content">
                      <p>{text}</p>
                      {text === 'Unavailable' && <p className="translation-unavailable">Translation not available</p>}
                    </div>
                  </div>
                ))}

                {/* Summaries */}
                {Object.entries(videoResults.summaries).map(([lang, text]) => (
                  <div key={lang} className="result-card">
                    <div className="result-card-header">
                      <h3>{getLanguageFlag(lang)} {getLanguageName(lang)} Summary</h3>
                      <DownloadButton content={text !== 'Unavailable' ? text : ''} language={lang} contentType="summary" token={authToken} />
                    </div>
                    <div className="scrollable-content">
                      <p>{text}</p>
                      {text === 'Unavailable' && <p className="translation-unavailable">Summary not available</p>}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </>

      ) : (

      /* ══════════════════════════════ STREAM TAB ════════════════════════════ */
        <>
          <section className="stream-section">
            <h2>Live Stream Processing</h2>
            <form className="stream-form" onSubmit={handleStreamSubmit}>
              <input
                type="text"
                className="stream-input"
                placeholder="Enter YouTube live stream URL"
                value={streamUrl}
                onChange={e => setStreamUrl(e.target.value)}
                disabled={isStreaming}
              />
              {isStreaming
                ? <button type="button" className="stream-button stop" onClick={handleStopStream}>Stop Processing</button>
                : <button type="submit"  className="stream-button"      disabled={!streamUrl}>Start Processing</button>
              }
            </form>

            {error && <div className="error-message">{error}</div>}

            <ProcessingStatus isStreaming={isStreaming} isUploading={false} status={connectionStatus} />
          </section>

          {/* ── Stream results ── */}
          {(isStreaming || streamResults.fullTranscript) && (
            <div className="results-section">
              {/* Header row */}
              <div className="result-header">
                <div className="result-header-left">
                  <h2>Live Stream Results</h2>
                  {streamResults.language && <span className="language-tag">{getLanguageName(streamResults.language)}</span>}
                </div>
                <div className="result-header-right">
                  {isStreaming && (
                    <div className="processing-indicator">
                      <div className="pulse-dot" />
                      <span>Processing live stream…</span>
                    </div>
                  )}

                  {/* Language selector dropdown */}
                  <div className="lang-selector-wrap" ref={dropdownRef}>
                    <button className="lang-selector-btn" onClick={() => setDropdownOpen(o => !o)}>
                      <span className="lang-selector-flag">{activeLang.flag}</span>
                      <span className="lang-selector-label">{activeLang.label}</span>
                      <svg className={`lang-selector-arrow ${dropdownOpen ? 'rotated' : ''}`} width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><polyline points="6 9 12 15 18 9"/></svg>
                    </button>
                    {dropdownOpen && (
                      <ul className="lang-dropdown" role="listbox" aria-label="Select language">
                        {LANGUAGES.map(lang => {
                          const hasContent = !!streamResults.liveTranslations[lang.key];
                          return (
                            <li
                              key={lang.key}
                              className={`lang-option ${selectedLang === lang.key ? 'selected' : ''} ${hasContent ? 'has-content' : ''}`}
                              role="option"
                              aria-selected={selectedLang === lang.key}
                              onClick={() => { setSelectedLang(lang.key); setDropdownOpen(false); }}
                            >
                              <span className="lang-option-flag">{lang.flag}</span>
                              <span className="lang-option-label">{lang.label}</span>
                              {hasContent && <span className="lang-option-dot" title="Content available" />}
                              {selectedLang === lang.key && (
                                <svg className="lang-option-check" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3"><polyline points="20 6 9 17 4 12"/></svg>
                              )}
                            </li>
                          );
                        })}
                      </ul>
                    )}
                  </div>
                </div>
              </div>

              {/* Two-column layout */}
              <div className="live-results-grid-improved">
                {/* LEFT: transcript */}
                <div className="live-transcript-card">
                  <div className="live-card-header">
                    <h3>Live Transcript ({getLanguageName(streamResults.language || 'en')})</h3>
                    <DownloadButton
                      content={streamResults.fullTranscript}
                      language={streamResults.language || 'en'}
                      contentType="transcript"
                      token={authToken}
                    />
                  </div>
                  <div className="live-transcript-content">
                    {streamResults.fullTranscript
                      ? <p className="flowing-transcript">{streamResults.fullTranscript}<span className="cursor-blink" /></p>
                      : <p className="empty-state">Waiting for transcription…</p>
                    }
                    <div ref={transcriptEndRef} />
                  </div>
                </div>

                {/* RIGHT: selected language translation */}
                <div className="live-translations-card">
                  <div className="live-card-header">
                    <h3>
                      <span className="live-translation-flag">{activeLang.flag}</span>
                      Translation — {activeLang.label}
                    </h3>
                    <DownloadButton
                      content={activeText}
                      language={selectedLang}
                      contentType="translation"
                      token={authToken}
                    />
                  </div>

                  <div className="live-translation-content">
                    {activeText
                      ? <p className="flowing-text">{activeText}<span className="cursor-blink-small" /></p>
                      : (
                        <div className="translation-waiting">
                          <div className="waiting-orb" />
                          <p className="empty-state-small">Waiting for {activeLang.label} translation…</p>
                        </div>
                      )
                    }
                  </div>

                  {/* Quick-switch pill strip */}
                  <div className="lang-availability-strip">
                    {LANGUAGES.map(lang => {
                      const ready = !!streamResults.liveTranslations[lang.key];
                      return (
                        <button
                          key={lang.key}
                          className={`lang-pill ${selectedLang === lang.key ? 'active' : ''} ${ready ? 'ready' : ''}`}
                          onClick={() => setSelectedLang(lang.key)}
                          title={`${lang.label}${ready ? ' — content ready' : ''}`}
                        >
                          <span>{lang.flag}</span>
                          <span className="lang-pill-code">{lang.label.slice(0, 3)}</span>
                          {ready && <span className="pill-ready-dot" />}
                        </button>
                      );
                    })}
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
