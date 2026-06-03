from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from eg_rsa.memory.memory_card import MemoryCard


class MemoryStore:
    """Simple JSONL memory store for EG-RSA.

    The first implementation intentionally avoids vector databases.  Retrieval is
    deterministic and easy to debug: cards are ranked by failure-mode overlap and
    optional environment-family match.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> List[MemoryCard]:
        if not self.path.exists():
            return []
        cards: List[MemoryCard] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cards.append(MemoryCard.from_dict(json.loads(line)))
        return cards

    def append(self, card: MemoryCard) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(card.to_dict(), ensure_ascii=False) + "\n")

    def retrieve(
        self,
        failure_modes: Iterable[str],
        env_family: str = "unknown",
        top_k: int = 3,
    ) -> List[MemoryCard]:
        query_modes = set(failure_modes)
        scored = []
        for card in self.load():
            card_modes = set(card.failure_modes)
            overlap = len(query_modes & card_modes)
            family_bonus = 1 if card.env_family == env_family else 0
            score = overlap * 10 + family_bonus
            if score > 0:
                scored.append((score, card))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [card for _, card in scored[:top_k]]
