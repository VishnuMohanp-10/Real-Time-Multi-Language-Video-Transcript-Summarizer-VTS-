import React from 'react';

const ProcessingStatus = ({ isStreaming, isUploading, status }) => {
  if (isStreaming) {
    return (
      <div className="processing-status">
        <div className="processing-indicator">
          <div className="pulse-dot"></div>
          <span>
            {status === 'connecting' ? 'Connecting to stream...' : 
             status === 'processing' ? 'Processing live stream...' : 
             'Waiting for stream data...'}
          </span>
        </div>
      </div>
    );
  }

  if (isUploading) {
    return (
      <div className="processing-status">
        <div className="processing-indicator">
          <div className="spinner"></div>
          <span>Processing video upload...</span>
        </div>
      </div>
    );
  }

  return null;
};

export default ProcessingStatus;
