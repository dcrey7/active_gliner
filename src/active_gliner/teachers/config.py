from pathlib import Path

import yaml
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[3]


class TeacherConfig(BaseModel):
    name: str
    base_url: str
    model: str
    api_key_env: str | None = None
    temperature: float = 0.0
    max_tokens: int = Field(default=512, gt=0)
    workers: int = Field(default=8, gt=0)
    gpu_shared: bool = False
    thinking: bool = False
    send_template_kwargs: bool = True
    json_schema: bool = True
    price_in_per_m: float = Field(default=0.0, ge=0)
    price_out_per_m: float = Field(default=0.0, ge=0)
    max_retries: int = Field(default=5, ge=0)
    retry_wait_s: float = Field(default=2.0, ge=0)
    max_wait_server_s: float = Field(default=900.0, ge=0)
    timeout_s: float = Field(default=120, gt=0)
    rpm_limit: int | None = Field(default=None, gt=0)
    pins: dict = Field(default_factory=dict)
    extra_body: dict = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TeacherConfig":
        return cls.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
