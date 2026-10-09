"""Translations of everything a player reads: the GUI, cards, wonder boards and the post-game analysis.

A language is a flat catalog that maps the **English text** to its translation
(``wonders/i18n/<lang>.json``; ``"_language"`` holds the language's own name). Extensions add
languages, or extend a bundled one with the names of their own cards, through
``registry.register_locale`` (see ``docs/extensions.md``). Anything missing falls back to English.

* Templates use :meth:`str.format` fields: ``"{n} Shields": "{n} Schilde"``. Numbers are formatted
  with the language's decimal separator (``"_decimal": ","``).
* A key may carry a context after ``|`` when one English word needs two translations
  (``"Loom"`` the card, ``"Loom|resource"`` the resource); lookups fall back to the bare word.
* Text built from parts is passed around as a :data:`Message` — ``{'text': template, 'args': {...}}``
  whose args may be messages (or lists of them, joined with ``", "``; ``{'join': '; ', 'items': [...]}``
  picks the separator) — so it can be stored in English and rendered later in any language (see
  :meth:`Translator.render`). Plain string args (player names) are never translated; wrap a name
  that should be in ``msg(name)``.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from typing import Any, Dict, List, Optional, Set

DEFAULT = 'en'
Message = Dict[str, Any]


def msg(text: str, context: Optional[str] = None, **args: Any) -> Message:
    """A translatable message: ``text`` is the English template, ``args`` fill its fields."""
    out: Message = {'text': text, 'args': args}
    if context:
        out['context'] = context
    return out


@lru_cache(maxsize=None)
def _bundled(lang: str) -> Dict[str, str]:
    path = resources.files('wonders.i18n').joinpath(f'{lang}.json')
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding='utf-8'))
    return {str(k): str(v) for k, v in data.items()}


def _bundled_languages() -> List[str]:
    return sorted(p.name[:-5] for p in resources.files('wonders.i18n').iterdir() if p.name.endswith('.json'))


def catalog(lang: str) -> Dict[str, str]:
    """Every translation of ``lang``: the bundled catalog, extended by extensions' locales."""
    from wonders.extensions import ensure_loaded

    registry = ensure_loaded()
    return {**_bundled(lang), **registry.locales.get(lang, {})}


def languages() -> Dict[str, str]:
    """Available languages, ``{code: own name}`` (English first)."""
    from wonders.extensions import ensure_loaded

    codes = dict.fromkeys([DEFAULT, *_bundled_languages(), *sorted(ensure_loaded().locales)])
    return {code: 'English' if code == DEFAULT else catalog(code).get('_language', code) for code in codes}


class _Number:
    """A number that formats with the language's decimal separator."""

    def __init__(self, value: Any, decimal: str):
        self.value, self.decimal = value, decimal

    def __format__(self, spec: str) -> str:
        text = format(self.value, spec)
        return text.replace('.', self.decimal) if self.decimal != '.' else text


class Translator:
    """Translate English text into one language (English itself is the identity).

    Args:
        lang: Language code; unknown languages translate nothing.
        missing: Collects the keys that had no translation (for tests and translators).
    """

    def __init__(self, lang: str = DEFAULT, missing: Optional[Set[str]] = None):
        self.lang = lang or DEFAULT
        self.missing = missing
        self._strings: Optional[Dict[str, str]] = None

    @property
    def strings(self) -> Dict[str, str]:
        if self._strings is None:
            self._strings = {} if self.lang == DEFAULT else catalog(self.lang)
        return self._strings

    def lookup(self, text: str, context: Optional[str] = None) -> str:
        """The translation of ``text`` (in ``context``, if given), or ``text`` itself."""
        strings = self.strings
        if context and f'{text}|{context}' in strings:
            return strings[f'{text}|{context}']
        if text in strings:
            return strings[text]
        if self.missing is not None and self.lang != DEFAULT and text:
            self.missing.add(f'{text}|{context}' if context else text)
        return text

    def __call__(self, text: str, context: Optional[str] = None, **args: Any) -> str:
        """Translate ``text`` and fill its fields; numbers get the language's decimal separator."""
        template = self.lookup(text, context)
        if not args:
            return template
        decimal = self.strings.get('_decimal', '.')
        values = {
            k: _Number(v, decimal) if isinstance(v, (int, float)) and not isinstance(v, bool) else v
            for k, v in args.items()
        }
        return template.format(**values)

    def render(self, message: Any) -> Any:
        """A :data:`Message` as text; anything else unchanged.

        A list of messages is joined with ``", "``, ``{'join': sep, 'items': [...]}`` with ``sep``.
        """
        if isinstance(message, list):
            return ', '.join(str(self.render(m)) for m in message)
        if isinstance(message, dict) and 'join' in message:
            return str(message['join']).join(str(self.render(m)) for m in message['items'])
        if isinstance(message, dict) and 'text' in message:
            args = {k: self.render(v) for k, v in message.get('args', {}).items()}
            return self(message['text'], message.get('context'), **args)
        return message

    def name(self, text: str) -> str:
        """A card, wonder or other proper name."""
        return self.lookup(text)


ENGLISH = Translator(DEFAULT)


def english(message: Any) -> str:
    """A :data:`Message` rendered in English."""
    return str(ENGLISH.render(message))


__all__ = ['DEFAULT', 'ENGLISH', 'Message', 'Translator', 'catalog', 'english', 'languages', 'msg']
