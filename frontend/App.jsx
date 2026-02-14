import React, { useState, useCallback, useEffect } from 'react';
import uploadIcon from './upload-icon.svg';
import DocumentAnimationIcon from './document-animation-icon.svg';

// Application states: 'idle', 'processing', 'success', 'error'

function App() {
  const [status, setStatus] = useState('idle');
  const [file, setFile] = useState(null);
  const [errorMessage, setErrorMessage] = useState('');
  const [dragOver, setDragOver] = useState(false);
  const [redactedFileUrl, setRedactedFileUrl] = useState(null);
  const [quirkyText, setQuirkyText] = useState('Redactifying...');
  const [redactedList, setRedactedList] = useState('');

  // --- Backend Integration Point ---
  // This is where you will connect to your Python/FastAPI backend.
  const API_ENDPOINT = 'http://localhost:8000/api/v1/redact';
  
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
    uploadAndProcessFile(selectedFile);
  };

  const uploadAndProcessFile = async (fileToProcess) => {
    const formData = new FormData();
    formData.append('pdf_file', fileToProcess);

    try {
      // ===================================================================
      // === REAL FETCH LOGIC (Comment out the MOCKED section to use) ====
      // ===================================================================
      // const response = await fetch(API_ENDPOINT, {
      //   method: 'POST',
      //   body: formData,
      // });

      // if (!response.ok) {
      //   const errorData = await response.json();
      //   throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      // }

      // const blob = await response.blob();
      // const url = URL.createObjectURL(blob);
      // setRedactedFileUrl(url);
      // setStatus('success');

      // ===================================================================
      // ============ MOCKED BEHAVIOR FOR UI DEMONSTRATION =================
      // ===================================================================
      console.log("Simulating file upload and processing...");
      await new Promise(resolve => setTimeout(resolve, 4000)); // Simulate network and processing time
      const mockBlob = new Blob(["This is a mock redacted PDF."], { type: 'application/pdf' });
      const mockUrl = URL.createObjectURL(mockBlob);
      setRedactedFileUrl(mockUrl);
      setStatus('success');

    } catch (error) {
      console.error("Processing failed:", error);
      setErrorMessage(error.message || 'An unknown error occurred during processing.');
      setStatus('error');
    }
  };

  const resetState = () => {
    setQuirkyText('Redactifying...');
    setStatus('idle');
    setFile(null);
    setErrorMessage('');
    if (redactedFileUrl) {
      URL.revokeObjectURL(redactedFileUrl);
    }
    setRedactedFileUrl(null);
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
      const phrases = ["Scanning for secrets...", "Applying PII-rifier...", "Thinking...", "Almost there..."];
      let i = 0;
      const interval = setInterval(() => {
        setQuirkyText(phrases[i]);
        i = (i + 1) % phrases.length;
      }, 2500);
      return () => clearInterval(interval);
    }
  }, [status]);

  // Effect for "typing out" the results on success
  useEffect(() => {
    if (status === 'success') {
      // This would come from the backend in a real app
      const mockResults = [
        "Redacted: SSN (***-**-****) on page 1.",
        "Redacted: Name (Tara Wilson) on page 1.",
        "Redacted: Address (123 Main St) on page 2.",
        "Redacted: Phone Number ((555) 123-4567) on page 2.",
        "Redacted: Employer (ACME Corp) on page 4."
      ];
      const fullText = "Redaction Report:\n" + mockResults.join('\n');
      let i = 0;
      setRedactedList(''); // Clear previous results
      const typingInterval = setInterval(() => {
        if (i < fullText.length) {
          setRedactedList(prev => prev + fullText.charAt(i));
          i++;
        } else {
          clearInterval(typingInterval);
          // Remove blinking cursor after typing is done
          const listElement = document.querySelector('.result-list');
          if(listElement) listElement.classList.add('done-typing');
        }
      }, 25); // Typing speed
      return () => clearInterval(typingInterval);
    }
  }, [status]);

  const renderLeftPane = () => (
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
              <p className="file-name-status">Processing: {file?.name}</p>
            </div>
          </div>
        );
      case 'success':
        return (
          <div className="split-layout">
            {renderLeftPane()}
            <div className="right-pane">
              <h2 className="result-header">Redaction Complete!</h2>
              <div className="result-list">{redactedList}</div>
              <div className="button-group">
                <a href={redactedFileUrl} download={`redacted_${file?.name}`} className="btn btn-primary">Download PDF</a>
                <button onClick={resetState} className="btn btn-secondary">Process Another</button>
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
      <header className="app-header">
        <h1>Aegis<span>.</span></h1>
        <p>
          Automatically detect and permanently remove sensitive information from your PDF documents.
        </p>
      </header>
      <main className="main-card">
        {renderContent()}
      </main>
    </div>
  );
}

export default App;