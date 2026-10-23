from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any


@dataclass(slots=True)
class E3Record:
    room: str
    algorithm: str
    initial_score: float
    best_score: float
    initial_feasible: bool
    best_feasible: bool
    evaluations: int
    runtime_s: float

    @property
    def delta(self) -> float:
        return float(self.best_score - self.initial_score)

    def to_dict(self) -> dict[str, Any]:
        return {
            "room": self.room,
            "algorithm": self.algorithm,
            "initial_score": round(self.initial_score, 4),
            "best_score": round(self.best_score, 4),
            "delta": round(self.delta, 4),
            "initial_feasible": self.initial_feasible,
            "best_feasible": self.best_feasible,
            "evaluations": self.evaluations,
            "runtime_s": round(self.runtime_s, 4),
        }


def summarize_e3(records: list[E3Record]) -> dict[str, Any]:
    algorithms = sorted({record.algorithm for record in records})
    summary: dict[str, Any] = {}
    for algorithm in algorithms:
        rows = [record for record in records if record.algorithm == algorithm]
        summary[algorithm] = {
            "rooms": len(rows),
            "mean_initial": round(mean(record.initial_score for record in rows), 4),
            "mean_best": round(mean(record.best_score for record in rows), 4),
            "mean_delta": round(mean(record.delta for record in rows), 4),
            "improved_rooms": sum(1 for record in rows if record.delta > 1e-9),
            "feasible_initial": sum(1 for record in rows if record.initial_feasible),
            "feasible_best": sum(1 for record in rows if record.best_feasible),
            "mean_runtime_s": round(mean(record.runtime_s for record in rows), 4),
            "total_evaluations": sum(record.evaluations for record in rows),
        }
    return {"rooms": len({record.room for record in records}), "algorithms": summary}


def format_e3_markdown(records: list[E3Record], summary: dict[str, Any] | None = None) -> str:
    summary = summary or summarize_e3(records)
    lines = [
        "# E3: Layout Optimization",
        "",
        "Initial = detection-only baseline; best = optimized layout.",
        "",
        "## Aggregate",
        "",
        "| Algorithm | Rooms | Mean initial | Mean best | Mean delta | Improved | Feasible init -> best | Mean runtime (s) | Evaluations |",
        "|---|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for algorithm, row in summary["algorithms"].items():
        lines.append(
            f"| {algorithm} | {row['rooms']} | {row['mean_initial']:.4f} | {row['mean_best']:.4f} | "
            f"{row['mean_delta']:+.4f} | {row['improved_rooms']}/{row['rooms']} | "
            f"{row['feasible_initial']} -> {row['feasible_best']} | {row['mean_runtime_s']:.3f} | "
            f"{row['total_evaluations']} |"
        )

    lines.extend(
        [
            "",
            "## Per room",
            "",
            "| Room | Algorithm | Initial | Best | Delta | Feasible | Runtime (s) | Evaluations |",
            "|---|---|---:|---:|---:|---|---:|---:|",
        ]
    )
    for record in sorted(records, key=lambda item: (item.room, item.algorithm)):
        lines.append(
            f"| {record.room} | {record.algorithm} | {record.initial_score:.4f} | "
            f"{record.best_score:.4f} | {record.delta:+.4f} | "
            f"{'yes' if record.best_feasible else 'no'} | {record.runtime_s:.3f} | "
            f"{record.evaluations} |"
        )
    return "\n".join(lines) + "\n"
