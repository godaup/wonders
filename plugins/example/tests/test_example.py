"""The plugin's own tests: run with ``pytest plugins/example`` after ``pip install plugins/example``."""

from pathlib import Path

from wonders.cards.content import load_content
from wonders.contrib import check_plugin
from wonders.extensions import ensure_loaded

PLUGIN = Path(__file__).resolve().parents[1]


def test_registers_effect_expansion_and_agent() -> None:
    registry = ensure_loaded()
    provides = registry.extensions['example'].provides
    assert provides == ['effect:tithe', 'expansion:tithe', 'agent:taxman']
    guild = load_content('base', 'tithe').card('Tax Collectors Guild')
    assert guild.guild and guild.expansion == 'tithe'


def test_passes_the_contribution_check() -> None:
    report = check_plugin(PLUGIN, games=2)
    assert report.passed, report.markdown()
