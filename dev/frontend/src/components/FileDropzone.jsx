import { useRef, useState } from "react";

export default function FileDropzone({
  onFileSelected,
  acceptedFormats,
  previewUrl,
  fileName,
  isAnalyzing = false,
  onClear
}) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef(null);

  const handleFile = (file) => {
    if (!file) return;
    onFileSelected(file);
  };

  const openFilePicker = () => {
    if (isAnalyzing) return;
    inputRef.current?.click();
  };

  return (
    <div
      className={dragging ? "dropzone dropzone-active" : "dropzone"}
      onDragOver={(event) => {
        if (isAnalyzing) return;
        event.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(event) => {
        if (isAnalyzing) return;
        event.preventDefault();
        setDragging(false);
        handleFile(event.dataTransfer.files?.[0]);
      }}
      onClick={openFilePicker}
    >
      <input
        ref={inputRef}
        type="file"
        className="hidden-input"
        accept={acceptedFormats.join(",")}
        disabled={isAnalyzing}
        onChange={(event) => handleFile(event.target.files?.[0])}
      />

      {previewUrl ? (
        <>
          <div className="dropzone-preview-frame">
            <img src={previewUrl} alt="Uploaded preview" className="dropzone-preview-image" />
            {isAnalyzing ? (
              <div className="scan-overlay" aria-hidden="true">
                <div className="scan-line" />
                <div className="scan-glow" />
              </div>
            ) : null}
          </div>

          <div className="dropzone-preview-meta">
            <p className="dropzone-file-name" title={fileName || "Selected image"}>
              {fileName || "Selected image"}
            </p>
            <div className="dropzone-actions">
              <button
                type="button"
                className="secondary-btn"
                onClick={(event) => {
                  event.stopPropagation();
                  openFilePicker();
                }}
              >
                Replace Image
              </button>
              <button
                type="button"
                className="ghost-btn"
                onClick={(event) => {
                  event.stopPropagation();
                  onClear?.();
                }}
                disabled={isAnalyzing}
              >
                Remove
              </button>
            </div>
          </div>
        </>
      ) : (
        <>
          <p className="dropzone-title">Drag & Drop Your Image Here</p>
          <p className="dropzone-subtitle">or click to browse your files</p>
          <p className="dropzone-formats">Supports: {acceptedFormats.join(", ")}</p>
          <button type="button" className="secondary-btn" style={{ marginTop: "16px" }}>
            Choose File
          </button>
        </>
      )}
    </div>
  );
}

