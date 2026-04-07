from __future__ import annotations


class AuthService:
	"""Temporary auth service used before database integration is added."""

	def __init__(self, enabled: bool) -> None:
		self.enabled = enabled

	def authenticate(self, email: str, password: str) -> tuple[bool, str, int]:
		_ = email
		_ = password

		if not self.enabled:
			return (
				False,
				"Sign-in is disabled. Database integration is not enabled yet.",
				503,
			)

		return (
			False,
			"Sign-in backend is not implemented yet. Database integration is the next milestone.",
			501,
		)

	def register(self, email: str, password: str) -> tuple[bool, str, int]:
		_ = email
		_ = password

		if not self.enabled:
			return (
				False,
				"Sign-up is disabled. Database integration is not enabled yet.",
				503,
			)

		return (
			False,
			"Sign-up backend is not implemented yet. Database integration is the next milestone.",
			501,
		)
