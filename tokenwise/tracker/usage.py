"""Track LLM usage and spending over time."""

import json
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path


@dataclass
class UsageEntry:
    """A single usage entry."""

    timestamp: str
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost": self.cost,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "UsageEntry":
        return cls(**data)


class UsageTracker:
    """Track and analyze LLM usage over time."""

    def __init__(self, storage_path: str | Path | None = None) -> None:
        self._entries: list[UsageEntry] = []
        self._path = Path(storage_path) if storage_path else None
        if self._path and self._path.exists():
            self._load()

    def log(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        tags: list[str] | None = None,
    ) -> UsageEntry:
        """Log a usage entry."""
        entry = UsageEntry(
            timestamp=datetime.now().isoformat(),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=round(cost, 6),
            tags=tags or [],
        )
        self._entries.append(entry)
        self._save()
        return entry

    def total_cost(self) -> float:
        """Total spending across all entries."""
        return round(sum(e.cost for e in self._entries), 6)

    def total_tokens(self) -> tuple[int, int]:
        """Total input and output tokens."""
        input_total = sum(e.input_tokens for e in self._entries)
        output_total = sum(e.output_tokens for e in self._entries)
        return input_total, output_total

    def cost_by_model(self) -> dict[str, float]:
        """Breakdown of spending per model."""
        breakdown: dict[str, float] = {}
        for e in self._entries:
            breakdown[e.model] = breakdown.get(e.model, 0) + e.cost
        return {
            k: round(v, 6) for k, v in sorted(breakdown.items(), key=lambda x: x[1], reverse=True)
        }

    def cost_by_day(self) -> dict[str, float]:
        """Breakdown of spending per day."""
        breakdown: dict[str, float] = {}
        for e in self._entries:
            day = e.timestamp[:10]
            breakdown[day] = breakdown.get(day, 0) + e.cost
        return {k: round(v, 6) for k, v in sorted(breakdown.items())}

    def entries(self) -> list[UsageEntry]:
        return list(self._entries)

    def _save(self) -> None:
        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            data = [e.to_dict() for e in self._entries]
            self._path.write_text(json.dumps(data, indent=2))

    def _load(self) -> None:
        if not self._path:
            return
        data = json.loads(self._path.read_text())
        self._entries = [UsageEntry.from_dict(d) for d in data]
