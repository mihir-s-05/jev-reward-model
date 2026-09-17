"""TypeSafe's documented POST /v1/systemone contract, with no invented SDK methods."""
from .config import ExperimentConfig
from .judge import HTTPJudge, parse_typesafe, questions


class JevClient(HTTPJudge):
    def __init__(self, cfg: ExperimentConfig, run_dir):
        super().__init__(cfg, run_dir, "jev")

    def payload(self, state: dict, kind: str) -> dict:
        return {"model": self.model, "state": state, "questions": questions(kind)}

    def parse(self, raw: dict, kind: str) -> dict:
        return parse_typesafe(raw, kind)
