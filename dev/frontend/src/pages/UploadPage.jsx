import { useEffect, useMemo, useState } from "react";
import FileDropzone from "@/components/FileDropzone";
import { analyzeImage, fetchHealth } from "@/api/client";
import { APP_CONFIG } from "@/constants";

function formatModelName(modelName = "") {
  const normalized = String(modelName).toLowerCase();
  if (!normalized) return "Unknown";
  if (normalized === "vit") return "ViT-B/16";
  if (normalized === "xception") return "Xception";
  if (normalized === "pg_fdd") return "PG-FDD";
  return modelName;
}

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

  const panelToneClass = useMemo(() => {
    if (!result) return "";
    return result.label === "FAKE" ? "analysis-pane-fake" : "analysis-pane-real";
  }, [result]);

  const displayProbability = useMemo(() => {
    if (!result) return 0;
    return result.label === "FAKE" ? result.fake_probability : 1 - result.fake_probability;
  }, [result]);

  const aiScore = useMemo(() => {
    if (!result) return 0;
    return result.fake_probability;
  }, [result]);

  const aiScoreToneClass = useMemo(() => {
    if (aiScore < 0.34) return "score-fill-safe";
    if (aiScore < 0.67) return "score-fill-caution";
    return "score-fill-risk";
  }, [aiScore]);

  const verdictSummary = useMemo(() => {
    if (!result) return "";
    return result.label === "FAKE" ? "Likely AI-generated profile image" : "Likely real profile image";
  }, [result]);

  const clearSelection = () => {
    setSelectedFile(null);
    setResult(null);
    setError("");
  };

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

        <div className="upload-workspace">
          <div className="workspace-left">
            <FileDropzone
              onFileSelected={setSelectedFile}
              acceptedFormats={APP_CONFIG.allowedFormats}
              previewUrl={previewUrl}
              fileName={selectedFile?.name || ""}
              isAnalyzing={loading}
              onClear={clearSelection}
            />
            <button
              className="primary-btn analyze-btn"
              type="button"
              onClick={handleAnalyze}
              disabled={loading || !selectedFile}
            >
              {loading ? "Analyzing Image..." : "Analyze Image"}
            </button>
            <p className="panel-note">
              Max upload size: {APP_CONFIG.maxUploadMb} MB. Images are processed in memory and not retained.
            </p>
          </div>

          <aside className={`analysis-pane ${panelToneClass}`.trim()}>
            <div className="analysis-pane-header">
              <h3>Analysis Result</h3>
              {result ? (
                <span className={result.label === "FAKE" ? "classification-pill pill-fake" : "classification-pill pill-real"}>
                  {result.label}
                </span>
              ) : null}
            </div>

            {loading ? (
              <div className="analysis-state analysis-state-loading">
                <div className="analysis-spinner" />
                <p>Scanning image patterns and extracting forensic signals...</p>
              </div>
            ) : null}

            {!loading && error ? <p className="error-text">{error}</p> : null}

            {!loading && !error && !result ? (
              <div className="analysis-state">
                <p>Upload an image and click Analyze Image to view the verdict and Grad-CAM explanation.</p>
              </div>
            ) : null}

            {!loading && !error && result ? (
              <div className="analysis-content">
                <div className={result.label === "FAKE" ? "verdict-banner verdict-banner-fake" : "verdict-banner verdict-banner-real"}>
                  <p className="verdict-label">{result.label}</p>
                  <p className="verdict-summary">{verdictSummary}</p>
                </div>

                <div className="analysis-metrics">
                  <div className="metric-tile">
                    <p className="metric-label">Confidence</p>
                    <p className="metric-value">{(result.confidence * 100).toFixed(2)}%</p>
                  </div>
                  <div className="metric-tile">
                    <p className="metric-label">Probability</p>
                    <p className="metric-value">{(displayProbability * 100).toFixed(2)}%</p>
                  </div>
                  <div className="metric-tile">
                    <p className="metric-label">Model</p>
                    <p className="metric-value">{formatModelName(result.model_name)}</p>
                  </div>
                </div>

                <div className="score-gauge">
                  <div className="score-gauge-row">
                    <span>AI Score</span>
                    <strong>{(aiScore * 100).toFixed(2)}%</strong>
                  </div>
                  <div className="score-track">
                    <div
                      className={`score-fill ${aiScoreToneClass}`.trim()}
                      style={{ width: `${Math.min(100, Math.max(0, aiScore * 100))}%` }}
                    />
                  </div>
                </div>

                <p className="result-explanation">{result.explanation}</p>

                {result.heatmap_overlay ? (
                  <div className="heatmap-panel">
                    <p className="heatmap-title">Grad-CAM Overlay</p>
                    <img src={result.heatmap_overlay} alt="Grad-CAM heatmap overlay" className="heatmap-image" />
                  </div>
                ) : null}
              </div>
            ) : null}
          </aside>
        </div>
      </section>
    </main>
  );
}

