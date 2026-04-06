import { useRef, useState } from "react";

export default function FileDropzone({ onFileSelected, acceptedFormats }) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef(null);

  const handleFile = (file) => {
    if (!file) return;
    onFileSelected(file);
  };

  return (
    <div
      className={dragging ? "dropzone dropzone-active" : "dropzone"}
      onDragOver={(event) => {
        event.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(event) => {
        event.preventDefault();
        setDragging(false);
        handleFile(event.dataTransfer.files?.[0]);
      }}
      onClick={() => inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        className="hidden-input"
        accept={acceptedFormats.join(",")}
        onChange={(event) => handleFile(event.target.files?.[0])}
      />
      <p className="dropzone-title">Drag & Drop Your Image Here</p>
      <p className="dropzone-subtitle">or click to browse your files</p>
      <p className="dropzone-formats">Supports: {acceptedFormats.join(", ")}</p>
      <button type="button" className="secondary-btn" style={{ marginTop: "16px" }}>
        Choose File
      </button>
    </div>
  );
}

