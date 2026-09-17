#!/usr/bin/env python
"""Tests for `wonders` package."""

import pytest
from click.testing import CliRunner

from wonders import cli


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string
    del response


def test_command_line_interface():
    """Test the CLI."""
    runner = CliRunner()
    result = runner.invoke(cli.main)
    assert result.exit_code == 0
    assert 'wonders' in result.output
    help_result = runner.invoke(cli.main, ['--help'])
    assert help_result.exit_code == 0
    assert '--help  Show this message and exit.' in help_result.output


def test_gallery_command_writes_html(tmp_path):
    """The gallery command renders cards + wonders to a self-contained HTML file."""
    out = tmp_path / 'g.html'
    result = CliRunner().invoke(cli.gallery, ['--out', str(out)])
    assert result.exit_code == 0
    html = out.read_text(encoding='utf-8')
    assert html.startswith('<!doctype html>')
    assert 'Wonder boards' in html and '<svg' in html


def test_table_command_plays_and_renders(tmp_path):
    """The table command plays a game and writes the final stats view."""
    out = tmp_path / 't.html'
    result = CliRunner().invoke(cli.table, ['--out', str(out), '--seed', '1'])
    assert result.exit_code == 0
    html = out.read_text(encoding='utf-8')
    assert html.startswith('<!doctype html>') and '<svg' in html
