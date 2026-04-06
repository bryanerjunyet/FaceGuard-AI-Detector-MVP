from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class FaceGuardSettings:
    app_name: str = "FaceGuard MVP"
    app_version: str = "0.1.0"

    # ── Active model ─────────────────────────────────────────────
    # Uncomment ONE block below, then restart the backend.
    #
    # ViT-B/16 (currently active):
    model_name: str = "vit"
    model_path: Path = REPO_ROOT / "models" / "pretrained" / "vit.pth"
    #
    # Xception:
    # model_name: str = "xception"
    # model_path: Path = REPO_ROOT / "models" / "pretrained" / "xception.pth"
    #
    # PG-FDD (Fair Deepfake Detector):
    # model_name: str = "pg_fdd"
    # model_path: Path = REPO_ROOT / "models" / "pretrained" / "pg_fdd.pth"
    #
    fake_threshold: float = 0.5

    # Upload policy
    max_upload_mb: int = 10
    allowed_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")
    allowed_mime_types: tuple[str, ...] = (
        "image/jpeg",
        "image/png",
        "image/webp",
    )

    # Frontend/backend local development
    backend_host: str = "127.0.0.1"
    backend_port: int = 8000
    frontend_origin: str = "http://localhost:5173"

    # Placeholder for future persistence
    enable_database: bool = False
    database_url: str = "mongodb://localhost:27017/faceguard"
    database_name: str = "faceguard"
    upload_collection: str = "uploads"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


SETTINGS = FaceGuardSettings()
