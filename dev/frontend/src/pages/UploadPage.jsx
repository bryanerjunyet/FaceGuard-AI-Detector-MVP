import { useEffect, useMemo, useState } from "react";
import FileDropzone from "@/components/FileDropzone";
import { analyzeImage, fetchHealth } from "@/api/client";
import { APP_CONFIG } from "@/constants";

export default function UploadPage() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [health, setHealth] = useState({ status: "checking", model_ready: false });

  useEffect(() => {
    fetchHealth()
      .then((payload) => setHealth({ status: "ok", ...payload }))
      .catch(() => setHealth({ status: "down", model_ready: false }));
  }, []);

  useEffect(() => {
    if (!selectedFile) {
      setPreviewUrl("");
      return;
    }
    const url = URL.createObjectURL(selectedFile);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [selectedFile]);

  const healthBadge = useMemo(() => {
    if (health.status === "checking") return "Backend: Checking...";
    if (health.status === "down") return "Backend: Offline";
    return health.model_ready ? "Model: Ready" : "Model: Not Loaded";
  }, [health]);

  const handleAnalyze = async () => {
    if (!selectedFile) {
      setError("Please select an image file for analysis.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);
    try {
      const payload = await analyzeImage(selectedFile);
      setResult(payload);
    } catch (requestError) {
      setError(requestError.message || "An unexpected error occurred during analysis.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="page">
      <section className="panel panel-upload">
        <div className="panel-header">
          <h2>Deepfake Detector</h2>
          <span className="status-pill">{healthBadge}</span>
        </div>

        <div className="upload-grid">
          <div>
            <FileDropzone onFileSelected={setSelectedFile} acceptedFormats={APP_CONFIG.allowedFormats} />
            <button className="primary-btn analyze-btn" type="button" onClick={handleAnalyze} disabled={loading}>
              {loading ? "Analyzing..." : "Analyze Image"}
            </button>
            <p className="panel-note">
              Max upload size: {APP_CONFIG.maxUploadMb} MB. Your privacy is respected; images are processed in memory
              and never stored.
            </p>
          </div>

          <div className="preview-panel">
            <p className="preview-title">Image Preview</p>
            {previewUrl ? (
              <img src={previewUrl} alt="Uploaded preview" className="preview-image" />
            ) : (
              <div className="preview-placeholder">Your selected image will appear here.</div>
            )}
          </div>
        </div>

        {error ? <p className="error-text">{error}</p> : null}

        {result ? (
          <div className={result.label === "FAKE" ? "result-card result-fake" : "result-card result-real"}>
            <p className="result-kicker">Analysis Complete</p>
            <p className="result-label-main">{result.label}</p>
            <p className="result-metric">
              <strong>Confidence:</strong> {(result.confidence * 100).toFixed(2)}%
            </p>
            <p className="result-metric">
              <strong>Fake Probability:</strong> {(result.fake_probability * 100).toFixed(2)}%
            </p>
            <p className="result-explanation">{result.explanation}</p>
          </div>
        ) : null}
      </section>
    </main>
  );
}

