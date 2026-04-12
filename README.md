# FaceGuard: a Privacy-Preserving AI for Detecting Deepfake Profile Photos

MCS24 Team is dedicated to develop **FaceGuard: a Privacy-Preserving AI for Detecting Deepfake Profile Photos**, a user-friendly web-based system to accurately identify and classify AI-generated human face images on social media and dating platforms. **FaceGuard** is anticipated to deliver a fully functional web-based system with visual explainability via Grad-CAM heatmaps that can accurately classify profile photos as real or AI-generated. 

> **NOTICE:** This repository is intended for Monash academic and MVP prototype use.

## Project Overview

The FaceGuard MVP provides:

- Web-based image upload and preview workflow
- Backend inference API for binary REAL/FAKE classification
- Confidence score and model metadata in responses
- In-memory processing with no database persistence
- Account authentication and authorisation
- Explainability overlays (Grad-CAM heatmaps)

## Repository Structure

```text
FaceGuard-AI-Detector/
├── config/
│   └── settings.py
│
├── dev/
│   ├── backend/
│   └── frontend/
│
├── docs/
│   ├── design/
│   ├── diagrams/
│   └── documents/
│
└── models/
  ├── pretrained/
  │   ├── vit.pth
  │   ├── xception.pth
  │   └── pg_fdd.pth
  │
  ├── README.md
  └── THIRD_PARTY_NOTICES.md
```

## Prerequisites

- Python 3.9+
- Node.js 18+
- npm 9+
- A compatible model checkpoint file (eg. `models/pretrained/vit.pth`)

## Quick Start

### 1. Install Python dependencies

From the repository root:

```bash
python -m venv .venv
```

Activate the virtual environment:

```bash
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

Install backend dependencies:

```bash
pip install -r requirements.txt
```

### 2. Install frontend dependencies

```bash
cd dev/frontend
npm install
```

### 3. Start backend

In a terminal from the repository root:

```bash
cd dev/backend
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### 4. Start frontend

In a second terminal:

```bash
cd dev/frontend
npm run dev
```

Open the application at:

- `http://localhost:5173`

## Health Endpoint

Verify backend readiness:

- `GET http://127.0.0.1:8000/api/health`

Example response:

```json
{
  "status": "ok",
  "model_ready": true,
  "model_name": "vit"
}
```

If `model_ready` is `false`, the backend is reachable but the configured
checkpoint file is missing or failed to load.

## API Contract

### GET /api/health

Returns service and model availability status.

### POST /api/analyze

Accepts `multipart/form-data` with field:

- `file` (JPEG, PNG, or WebP)

Example response:

```json
{
  "label": "FAKE",
  "confidence": 0.9123,
  "fake_probability": 0.9123,
  "threshold": 0.5,
  "explanation": "Prediction is based on a ViT deepfake classifier...",
  "model_name": "vit",
  "heatmap_overlay": "data:image/png;base64,...",
  "explainability_method": "grad_cam"
}
```

## Model Checkpoints

The backend loads model checkpoints from `models/pretrained/` based on
`config/settings.py`.

Common checkpoint names:

- `vit.pth`
- `xception.pth`
- `pg_fdd.pth`

Large model binaries are excluded from source control. Keep checkpoints in local
or approved storage and distribute them according to license terms.

## Design and Project Artifacts

Supporting design and project materials are available in:

- `docs/design/`
- `docs/diagrams/`
- `docs/documents/`

## Additional Documentation

For development operations and troubleshooting guidance, refer to [**Developer Guide**](DEVELOPER_GUIDE.md).

## Third-Party Notices

Model and dataset provenance information is maintained in [**Third-Party Notices**](models/THIRD_PARTY_NOTICES.md).
