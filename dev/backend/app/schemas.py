from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
	status: str = "ok"
	model_ready: bool
	model_name: str


class PredictionResponse(BaseModel):
	label: str = Field(description="REAL or FAKE")
	confidence: float = Field(ge=0.0, le=1.0)
	fake_probability: float = Field(ge=0.0, le=1.0)
	threshold: float = Field(ge=0.0, le=1.0)
	explanation: str
	model_name: str


class SignInRequest(BaseModel):
	email: str = Field(default="", max_length=320)
	password: str = Field(default="", max_length=128)


class SignInResponse(BaseModel):
	success: bool = True
	message: str


class SignUpRequest(BaseModel):
	email: str = Field(default="", max_length=320)
	password: str = Field(default="", max_length=128)


class SignUpResponse(BaseModel):
	success: bool = True
	message: str
