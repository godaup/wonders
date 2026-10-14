"""A fixed, discrete action space over the engine's variable legal-action lists.

Neural policies need a constant-size output. An action is identified by *what* is
done to *which card name* — ``(card name, ActionKind)`` — so action ``i`` means the
same thing in every state, independent of hand order. Two copies of a card in one
hand are interchangeable, so they share an index. The legal subset at a decision is
exposed as a boolean mask.

With Armada (``naval=True``) the space has a second block of the same size: the same actions with a Naval
Construction (``Action.naval``), at ``n_base + card * 4 + kind``.
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

    def __init__(self, card_names: Sequence[str], naval: bool = False):
        self.card_names: List[str] = sorted(set(card_names))
        self._card_index: Dict[str, int] = {name: i for i, name in enumerate(self.card_names)}
        self._kind_index: Dict[ActionKind, int] = {kind: i for i, kind in enumerate(KINDS)}
        self.naval = naval  # Armada: a second block for the actions with a Naval Construction

    @classmethod
    def from_content(cls, content: Content, naval: bool = False) -> 'ActionSpace':
        return cls([card.name for card in content.choices()], naval)

    @property
    def n_base(self) -> int:
        """The size of the block without Naval Constructions."""
        return len(self.card_names) * len(KINDS)

    @property
    def n(self) -> int:
        return self.n_base * (2 if self.naval else 1)

    def index(self, action: Action) -> int:
        index = self._card_index[action.card.name] * len(KINDS) + self._kind_index[action.kind]
        if action.naval:
            if not self.naval:
                raise ValueError('this action space has no Naval Constructions (Armada)')
            index += self.n_base
        return index

    def describe(self, index: int) -> Tuple[str, ActionKind]:
        """The ``(card name, kind)`` an index stands for (``is_naval`` tells the two blocks apart)."""
        card, kind = divmod(index % self.n_base, len(KINDS))
        return self.card_names[card], KINDS[kind]

    def is_naval(self, index: int) -> bool:
        return index >= self.n_base

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
