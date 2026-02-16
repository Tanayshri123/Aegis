import React, { useState, useCallback, useEffect, useRef } from 'react';
import uploadIcon from './upload-icon.svg';
import DocumentAnimationIcon from './document-animation-icon.svg';

// Application states: 'idle', 'processing', 'success', 'error'

function App() {
  const [status, setStatus] = useState('idle');
  const [file, setFile] = useState(null);
  const [errorMessage, setErrorMessage] = useState('');
  const [dragOver, setDragOver] = useState(false);
  const [quirkyText, setQuirkyText] = useState('Redactifying...');

  // Job data from backend
  const [jobId, setJobId] = useState(null);
  const [pageCount, setPageCount] = useState(0);
  const [entities, setEntities] = useState([]);

  // Pipeline progress
  const [pipelineStep, setPipelineStep] = useState(0);
  const [stepDetail, setStepDetail] = useState('');
  const pollRef = useRef(null);

  // Comparison modal
  const [showCompare, setShowCompare] = useState(false);

  // Redaction details modal
  const [showRedactionDetails, setShowRedactionDetails] = useState(false);

  // Chat state
  const [chatMessages, setChatMessages] = useState([]);
  const [chatInput, setChatInput] = useState('');
  const [chatLoading, setChatLoading] = useState(false);
  const [pendingRedaction, setPendingRedaction] = useState(null);
  const [pageRefreshKey, setPageRefreshKey] = useState(0);
  const chatEndRef = useRef(null);

  // Suggestion chips visibility
  const [showSuggestions, setShowSuggestions] = useState(true);

  // Undo/redo tracking
  const [redactionHistoryCount, setRedactionHistoryCount] = useState(0);
  const [redoAvailable, setRedoAvailable] = useState(false);

  const stopPolling = () => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  };

  const startPolling = (id) => {
    stopPolling();
    pollRef.current = setInterval(async () => {
      try {
        const res = await fetch(`/api/v1/jobs/${id}/progress`);
        if (!res.ok) return;
        const data = await res.json();

        setPipelineStep(data.step);
        setStepDetail(data.step_detail);

        if (data.status === 'done') {
          stopPolling();
          setPageCount(data.page_count);
          setEntities(data.entities || []);
          setStatus('success');
        } else if (data.status === 'error') {
          stopPolling();
          setErrorMessage(data.error || 'Pipeline failed.');
          setStatus('error');
        }
      } catch {
        // network hiccup, keep polling
      }
    }, 1500);
  };

  // Cleanup polling on unmount
  useEffect(() => {
    return () => stopPolling();
  }, []);

  const handleFileChange = (selectedFile) => {
    if (!selectedFile) return;

    if (selectedFile.type !== 'application/pdf') {
      setErrorMessage('Only PDF files are accepted.');
      setStatus('error');
      return;
    }

    setFile(selectedFile);
    setStatus('processing');
    setErrorMessage('');
    setPipelineStep(0);
    setStepDetail('Uploading...');
    uploadAndProcessFile(selectedFile);
  };

  const uploadAndProcessFile = async (fileToProcess) => {
    const formData = new FormData();
    formData.append('pdf_file', fileToProcess);

    try {
      const response = await fetch('/api/v1/redact', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setJobId(data.job_id);

      // Start polling for pipeline progress
      startPolling(data.job_id);

    } catch (error) {
      console.error("Upload failed:", error);
      setErrorMessage(error.message || 'An unknown error occurred.');
      setStatus('error');
    }
  };

  const resetState = () => {
    stopPolling();
    setQuirkyText('Redactifying...');
    setStatus('idle');
    setFile(null);
    setErrorMessage('');
    setJobId(null);
    setPageCount(0);
    setEntities([]);
    setShowCompare(false);
    setShowRedactionDetails(false);
    setPipelineStep(0);
    setStepDetail('');
    setChatMessages([]);
    setChatInput('');
    setChatLoading(false);
    setPendingRedaction(null);
    setPageRefreshKey(0);
    setShowSuggestions(true);
    setRedactionHistoryCount(0);
    setRedoAvailable(false);
  };

  const onDragOver = useCallback((e) => {
    e.preventDefault();
    setDragOver(true);
  }, []);

  const onDragLeave = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
  }, []);

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileChange(e.dataTransfer.files[0]);
      e.dataTransfer.clearData();
    }
  }, []);

  // Effect for cycling quirky phrases during processing
  useEffect(() => {
    if (status === 'processing') {
      const phrases = ["Scanning for secrets...", "Applying PII remove-ifier...", "Thinking...", "Almost there...", "Hunting down sensitive info...", "Redactifying...", "Scrubbing...",];
      let i = 0;
      const interval = setInterval(() => {
        setQuirkyText(phrases[i]);
        i = (i + 1) % phrases.length;
      }, 2500);
      return () => clearInterval(interval);
    }
  }, [status]);

  // Auto-greeting when chat becomes available
  useEffect(() => {
    if (status === 'success' && chatMessages.length === 0) {
      setChatMessages([{
        role: 'assistant',
        content: `Redaction complete! I found and redacted ${entities.length} PII entit${entities.length === 1 ? 'y' : 'ies'}. Ask me questions about the document or request additional redactions — e.g., "Also redact all mentions of Company X."`
      }]);
    }
  }, [status]);

  // Auto-scroll chat to bottom
  useEffect(() => {
    if (chatEndRef.current) {
      chatEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [chatMessages, chatLoading]);

  const sendChatMessage = async (messageOverride) => {
    const userMessage = (messageOverride || chatInput).trim();
    if (!userMessage || chatLoading) return;

    if (!messageOverride) setChatInput('');
    setShowSuggestions(false);
    setChatMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setChatLoading(true);

    try {
      const res = await fetch(`/api/v1/jobs/${jobId}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMessage }),
      });

      const data = await res.json();

      const newMsg = { role: 'assistant', content: data.reply };
      if (data.pdf_updated) {
        newMsg.isRedaction = true;
      }

      setChatMessages(prev => [...prev, newMsg]);

      if (data.redaction_request) {
        setPendingRedaction(data.redaction_request);
      }

      if (data.pdf_updated) {
        setPageRefreshKey(prev => prev + 1);
        setRedactionHistoryCount(prev => prev + 1);
        setRedoAvailable(false);
        // Re-fetch entities from progress endpoint
        const progRes = await fetch(`/api/v1/jobs/${jobId}/progress`);
        if (progRes.ok) {
          const progData = await progRes.json();
          setEntities(progData.entities || []);
          setPageCount(progData.page_count);
        }
      }
    } catch (err) {
      setChatMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Sorry, something went wrong. Please try again.'
      }]);
    } finally {
      setChatLoading(false);
    }
  };

  const handleConfirmRedaction = async () => {
    if (!pendingRedaction) return;

    const terms = pendingRedaction.terms;
    setPendingRedaction(null);
    setChatLoading(true);
    setChatMessages(prev => [...prev, {
      role: 'user',
      content: `Confirmed: redact ${terms.map(t => `"${t}"`).join(', ')}`
    }]);

    try {
      const res = await fetch(`/api/v1/jobs/${jobId}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: 'confirm_redaction', confirmed_terms: terms }),
      });

      const data = await res.json();

      const newMsg = { role: 'assistant', content: data.reply };
      if (data.pdf_updated) {
        newMsg.isRedaction = true;
      }
      setChatMessages(prev => [...prev, newMsg]);

      if (data.pdf_updated) {
        setPageRefreshKey(prev => prev + 1);
        setRedactionHistoryCount(prev => prev + 1);
        setRedoAvailable(false);
        const progRes = await fetch(`/api/v1/jobs/${jobId}/progress`);
        if (progRes.ok) {
          const progData = await progRes.json();
          setEntities(progData.entities || []);
          setPageCount(progData.page_count);
        }
      }
    } catch (err) {
      setChatMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Redaction failed. Please try again.'
      }]);
    } finally {
      setChatLoading(false);
    }
  };

  const handleCancelRedaction = () => {
    setPendingRedaction(null);
    setChatMessages(prev => [...prev, {
      role: 'assistant',
      content: 'Redaction cancelled. Let me know if you need anything else.'
    }]);
  };

  const handleChatKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendChatMessage();
    }
  };

  const handleUndo = async (msgIndex) => {
    if (chatLoading) return;
    setChatLoading(true);

    try {
      const res = await fetch(`/api/v1/jobs/${jobId}/undo`, { method: 'POST' });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || 'Undo failed');
      }

      const data = await res.json();
      setEntities(data.entities || []);
      setPageCount(data.page_count);
      setPageRefreshKey(prev => prev + 1);
      setRedactionHistoryCount(prev => Math.max(0, prev - 1));
      setRedoAvailable(true);

      // Mark the redaction message as undone
      setChatMessages(prev => {
        const updated = [...prev];
        if (updated[msgIndex]) {
          updated[msgIndex] = { ...updated[msgIndex], isRedaction: false, undone: true };
        }
        updated.push({ role: 'assistant', content: 'Redaction undone. Click Redo to re-apply.' });
        return updated;
      });
    } catch (err) {
      setChatMessages(prev => [...prev, {
        role: 'assistant',
        content: `Undo failed: ${err.message}`
      }]);
    } finally {
      setChatLoading(false);
    }
  };

  const handleRedo = async (msgIndex) => {
    if (chatLoading) return;
    setChatLoading(true);

    try {
      const res = await fetch(`/api/v1/jobs/${jobId}/redo`, { method: 'POST' });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || 'Redo failed');
      }

      const data = await res.json();
      setEntities(data.entities || []);
      setPageCount(data.page_count);
      setPageRefreshKey(prev => prev + 1);
      setRedactionHistoryCount(prev => prev + 1);
      setRedoAvailable(false);

      // Restore the redaction message
      setChatMessages(prev => {
        const updated = [...prev];
        if (updated[msgIndex]) {
          updated[msgIndex] = { ...updated[msgIndex], isRedaction: true, undone: false };
        }
        updated.push({ role: 'assistant', content: 'Redaction re-applied.' });
        return updated;
      });
    } catch (err) {
      setChatMessages(prev => [...prev, {
        role: 'assistant',
        content: `Redo failed: ${err.message}`
      }]);
    } finally {
      setChatLoading(false);
    }
  };

  // Find the index of the last redaction message (for showing undo button)
  const lastRedactionIndex = (() => {
    for (let i = chatMessages.length - 1; i >= 0; i--) {
      if (chatMessages[i].isRedaction) return i;
    }
    return -1;
  })();

  // Find the last undone message (for showing redo button)
  const lastUndoneIndex = (() => {
    for (let i = chatMessages.length - 1; i >= 0; i--) {
      if (chatMessages[i].undone) return i;
    }
    return -1;
  })();

  const renderLeftPane = () => {
    // After redaction success: show actual redacted PDF pages (with cache-busting)
    if (status === 'success' && jobId && pageCount > 0) {
      const pageImages = [];
      for (let i = 0; i < pageCount; i++) {
        pageImages.push(
          <img
            key={`${i}-${pageRefreshKey}`}
            src={`/api/v1/jobs/${jobId}/pages/${i}?type=redacted&v=${pageRefreshKey}`}
            alt={`Redacted page ${i + 1}`}
            className="pdf-page-image"
          />
        );
      }
      return (
        <div className="left-pane">
          <div className="pdf-viewer">
            {pageImages}
          </div>
        </div>
      );
    }

    // During processing: show animation
    return (
      <div className="left-pane">
        <div className="document-anim-container">
          <img src={DocumentAnimationIcon} alt="Document Animation" />
          {status === 'processing' && (
            <>
              <div className="redaction-bar"></div>
              <div className="redaction-bar"></div>
              <div className="redaction-bar"></div>
            </>
          )}
        </div>
      </div>
    );
  };

  const renderCompareModal = () => {
    if (!showCompare || !jobId) return null;

    const originalPages = [];
    const redactedPages = [];
    for (let i = 0; i < pageCount; i++) {
      originalPages.push(
        <img
          key={`orig-${i}`}
          src={`/api/v1/jobs/${jobId}/pages/${i}?type=original`}
          alt={`Original page ${i + 1}`}
          className="compare-page-image"
        />
      );
      redactedPages.push(
        <img
          key={`redact-${i}`}
          src={`/api/v1/jobs/${jobId}/pages/${i}?type=redacted`}
          alt={`Redacted page ${i + 1}`}
          className="compare-page-image"
        />
      );
    }

    return (
      <div className="modal-overlay" onClick={() => setShowCompare(false)}>
        <div className="modal-content" onClick={(e) => e.stopPropagation()}>
          <div className="modal-header">
            <h2>Compare Redaction</h2>
            <button className="modal-close" onClick={() => setShowCompare(false)}>&times;</button>
          </div>
          <div className="compare-container">
            <div className="compare-column">
              <h3>Original</h3>
              <div className="compare-scroll">
                {originalPages}
              </div>
            </div>
            <div className="compare-divider"></div>
            <div className="compare-column">
              <h3>Redacted</h3>
              <div className="compare-scroll">
                {redactedPages}
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  const renderRedactionDetailsModal = () => {
    if (!showRedactionDetails) return null;

    // Group entities by type
    const grouped = {};
    for (const e of entities) {
      if (!grouped[e.type]) grouped[e.type] = [];
      grouped[e.type].push(e);
    }

    return (
      <div className="modal-overlay" onClick={() => setShowRedactionDetails(false)}>
        <div className="modal-content redaction-details-modal" onClick={(e) => e.stopPropagation()}>
          <div className="modal-header">
            <h2>Redactions Made</h2>
            <button className="modal-close" onClick={() => setShowRedactionDetails(false)}>&times;</button>
          </div>
          <div className="redaction-details-body">
            {Object.keys(grouped).length === 0 ? (
              <p className="no-redactions">No redactions have been applied yet.</p>
            ) : (
              Object.entries(grouped).sort(([a], [b]) => a.localeCompare(b)).map(([type, ents]) => (
                <div key={type} className="redaction-type-group">
                  <div className="redaction-type-header">
                    <span className="redaction-type-badge">{type.replace(/_/g, ' ')}</span>
                    <span className="redaction-type-count">{ents.length} instance{ents.length !== 1 ? 's' : ''}</span>
                  </div>
                  <ul className="redaction-instances">
                    {ents.map((e, idx) => (
                      <li key={idx} className="redaction-instance">
                        <span className="redaction-value">"{e.value}"</span>
                        <span className="redaction-page">page {e.page}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    );
  };

  const renderProgressTracker = () => {
    const steps = [
      { num: 1, label: "Extract Text" },
      { num: 2, label: "Detect PII" },
      { num: 3, label: "Locate Bboxes" },
      { num: 4, label: "Redact PDF" },
    ];

    // Progress bar percentage
    const progressPercent = Math.min(100, (pipelineStep / 4) * 100);

    return (
      <>
        <div className="progress-tracker">
          {steps.map((s) => {
            let cls = "progress-step";
            if (pipelineStep > s.num) cls += " completed";
            else if (pipelineStep === s.num) cls += " active";
            return (
              <div key={s.num} className={cls}>
                <div className="step-indicator">
                  {pipelineStep > s.num ? (
                    <span className="step-check">&#10003;</span>
                  ) : (
                    <span className="step-num">{s.num}</span>
                  )}
                </div>
                <span className="step-label">{s.label}</span>
              </div>
            );
          })}
        </div>
        <div className="progress-bar-container">
          <div className="progress-bar-fill" style={{ width: `${progressPercent}%` }}></div>
        </div>
      </>
    );
  };

  const suggestionChips = [
    { label: "What's left?", message: "What's left to redact?" },
    { label: "Redact SSNs", message: "Redact all SSNs" },
    { label: "Redact names", message: "Redact all names" },
    { label: "Redact addresses", message: "Redact all addresses" },
    { label: "Redaction summary", message: "Show me what was redacted" },
    { label: "Page 1 info", message: "What's on page 1?" },
  ];

  const renderContent = () => {
    switch (status) {
      case 'processing':
        return (
          <div className="split-layout">
            {renderLeftPane()}
            <div className="right-pane">
              <div className="quirky-text-container">
                <div className="mini-bubble"></div>
                <div className="mini-bubble"></div>
                <div className="mini-bubble"></div>
                <h2 className="quirky-text">{quirkyText}</h2>
              </div>
              {renderProgressTracker()}
              <p className="step-detail-text">{stepDetail}</p>
              <p className="file-name-status">Processing: {file?.name}</p>
            </div>
          </div>
        );
      case 'success':
        return (
          <div className="split-layout">
            {renderLeftPane()}
            <div className="right-pane right-pane-chat">
              {/* Chat header with branding */}
              <div className="chat-header-brand">
                <h1 className="chat-brand">Aegis<span>.</span></h1>
              </div>

              {/* Compact summary bar - sticky */}
              <div className="summary-bar">
                <div className="summary-info" onClick={() => setShowRedactionDetails(true)} title="Click to view redaction details">
                  <span className="summary-check">&#10003;</span>
                  <span className="summary-text">
                    Redacted {entities.length} entit{entities.length === 1 ? 'y' : 'ies'}
                  </span>
                  <span className="summary-expand-icon">&#9662;</span>
                </div>
                <div className="summary-actions">
                  <a
                    href={`/api/v1/jobs/${jobId}/download`}
                    download={`redacted_${file?.name}`}
                    className="btn btn-sm btn-primary"
                  >
                    Download
                  </a>
                  <button onClick={() => setShowCompare(true)} className="btn btn-sm btn-compare">
                    Compare
                  </button>
                  <button onClick={resetState} className="btn btn-sm btn-secondary">
                    New File
                  </button>
                </div>
              </div>

              {/* Chat panel */}
              <div className="chat-container">
                <div className="chat-messages">
                  {chatMessages.map((msg, idx) => (
                    <div key={idx} className={`chat-message ${msg.role}${msg.undone ? ' undone' : ''}`}>
                      <span>{msg.content}</span>
                      {msg.isRedaction && idx === lastRedactionIndex && redactionHistoryCount > 0 && (
                        <button
                          className="undo-btn"
                          onClick={() => handleUndo(idx)}
                          disabled={chatLoading}
                          title="Undo this redaction"
                        >
                          &#x21A9; Undo
                        </button>
                      )}
                      {msg.undone && idx === lastUndoneIndex && redoAvailable && (
                        <button
                          className="redo-btn"
                          onClick={() => handleRedo(idx)}
                          disabled={chatLoading}
                          title="Redo this redaction"
                        >
                          &#x21AA; Redo
                        </button>
                      )}
                    </div>
                  ))}

                  {/* Pending redaction confirmation */}
                  {pendingRedaction && (
                    <div className="chat-confirm-bar">
                      <p>Redact: {pendingRedaction.terms.map(t => `"${t}"`).join(', ')}?</p>
                      <div className="chat-confirm-actions">
                        <button onClick={handleConfirmRedaction} className="btn btn-sm btn-primary" disabled={chatLoading}>
                          Confirm
                        </button>
                        <button onClick={handleCancelRedaction} className="btn btn-sm btn-secondary" disabled={chatLoading}>
                          Cancel
                        </button>
                      </div>
                    </div>
                  )}

                  {/* Loading indicator */}
                  {chatLoading && (
                    <div className="chat-message assistant">
                      <div className="typing-dots">
                        <span></span><span></span><span></span>
                      </div>
                    </div>
                  )}

                  <div ref={chatEndRef} />
                </div>

                {/* Suggestion chips - always available, toggleable */}
                <div className="suggestions-area">
                  <button
                    className="suggestions-toggle"
                    onClick={() => setShowSuggestions(prev => !prev)}
                  >
                    Quick actions {showSuggestions ? '\u25B4' : '\u25BE'}
                  </button>
                  {showSuggestions && (
                    <>
                      <div className="suggestions-info">
                        Try asking me to redact specific terms, entire PII categories, or explore document contents.
                      </div>
                      <div className="suggestions-chips">
                        {suggestionChips.map((chip, idx) => (
                          <button
                            key={idx}
                            className="suggestion-chip"
                            onClick={() => sendChatMessage(chip.message)}
                            disabled={chatLoading}
                          >
                            {chip.label}
                          </button>
                        ))}
                      </div>
                    </>
                  )}
                </div>

                <div className="chat-input-bar">
                  <input
                    type="text"
                    className="chat-input"
                    placeholder="Ask about the document or request redactions..."
                    value={chatInput}
                    onChange={(e) => setChatInput(e.target.value)}
                    onKeyDown={handleChatKeyDown}
                    disabled={chatLoading}
                  />
                  <button
                    className="chat-send-btn"
                    onClick={() => sendChatMessage()}
                    disabled={chatLoading || !chatInput.trim()}
                  >
                    &#x27A4;
                  </button>
                </div>
              </div>
            </div>
          </div>
        );
      case 'error':
        return (
          <div className="split-layout">
            {renderLeftPane()}
            <div className="right-pane">
              <h2 className="result-header" style={{color: 'var(--color-error)'}}>Processing Failed</h2>
              <p style={{color: 'var(--color-text-muted)', marginBottom: '1.5rem'}}>{errorMessage}</p>
              <button onClick={resetState} className="btn btn-secondary">Try Again</button>
            </div>
          </div>
        );
      case 'idle':
      default:
        return (
          <div className="idle-view">
            <div
              className={`upload-area ${dragOver ? 'drag-over' : ''}`}
              onDragOver={onDragOver}
              onDragLeave={onDragLeave}
              onDrop={onDrop}
              onClick={() => document.getElementById('file-input').click()}
            >
              <img src={uploadIcon} alt="Upload" className="upload-icon" />
              <div className="upload-text">
                <p><strong>Click to upload</strong> or drag and drop</p>
                <p>PDF files only</p>
              </div>
              <input
                type="file"
                id="file-input"
                style={{ display: 'none' }}
                accept="application/pdf"
                onChange={(e) => handleFileChange(e.target.files[0])}
              />
            </div>
          </div>
        );
    }
  };

  return (
    <div className="app-container">
      <div className="background-bubbles">
        <div className="bubble bubble-1"></div>
        <div className="bubble bubble-2"></div>
        <div className="bubble bubble-3"></div>
        <div className="bubble bubble-4"></div>
        <div className="bubble bubble-5"></div>
        <div className="bubble bubble-6"></div>
      </div>
      {status !== 'success' && (
        <header className="app-header">
          <h1>Aegis<span>.</span></h1>
          <p>
            Automatically detect and permanently remove sensitive information from your PDF documents.
          </p>
        </header>
      )}
      <main className="main-card">
        {renderContent()}
      </main>
      {renderCompareModal()}
      {renderRedactionDetailsModal()}
    </div>
  );
}

export default App;
