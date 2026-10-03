"""A fixed, discrete action space over the engine's variable legal-action lists.

Neural policies need a constant-size output. An action is identified by *what* is
done to *which card name* — ``(card name, ActionKind)`` — so action ``i`` means the
same thing in every state, independent of hand order. Two copies of a card in one
hand are interchangeable, so they share an index. The legal subset at a decision is
exposed as a boolean mask.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from wonders.cards.content import Content
from wonders.engine.actions import Action, ActionKind

KINDS: Tuple[ActionKind, ...] = (ActionKind.BUILD, ActionKind.WONDER, ActionKind.DISCARD, ActionKind.FREE)


class ActionSpace:
    """Maps :class:`Action` objects to integers in ``range(n)`` and back.

    Index layout: ``card_index * len(KINDS) + kind_index``.
    """

    def __init__(self, card_names: Sequence[str]):
        self.card_names: List[str] = sorted(set(card_names))
        self._card_index: Dict[str, int] = {name: i for i, name in enumerate(self.card_names)}
        self._kind_index: Dict[ActionKind, int] = {kind: i for i, kind in enumerate(KINDS)}

    @classmethod
    def from_content(cls, content: Content) -> 'ActionSpace':
        return cls([card.name for card in content.cards])

    @property
    def n(self) -> int:
        return len(self.card_names) * len(KINDS)

    def index(self, action: Action) -> int:
        return self._card_index[action.card.name] * len(KINDS) + self._kind_index[action.kind]

    def describe(self, index: int) -> Tuple[str, ActionKind]:
        """The ``(card name, kind)`` an index stands for."""
        card, kind = divmod(index, len(KINDS))
        return self.card_names[card], KINDS[kind]

    def mask(self, legal: Sequence[Action]) -> List[bool]:
        """Boolean mask over ``range(n)``: True where an index is legal."""
        out = [False] * self.n
        for action in legal:
            out[self.index(action)] = True
        return out

    def to_action(self, index: int, legal: Sequence[Action]) -> Action:
        """Resolve ``index`` to the concrete legal :class:`Action` it denotes.

        Raises ``ValueError`` if the index is not legal at this decision.
        """
        for action in legal:
            if self.index(action) == index:
                return action
        name, kind = self.describe(index)
        raise ValueError(f'illegal action {index} ({kind.value} {name!r})')
