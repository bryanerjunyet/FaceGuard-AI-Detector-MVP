from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StoragePlaceholder:
	"""
	Placeholder for future persistence integration.
	Database is intentionally disabled in this MVP.
	"""

	enabled: bool = False

	def save_inference_event(self, payload: dict) -> None:  # noqa: ARG002
		if self.enabled:
			raise NotImplementedError("Database integration is not implemented in MVP.")
		# Intentionally no-op for local privacy-first MVP.
		return
