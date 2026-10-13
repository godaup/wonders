"""Tests for local human play: the session layer and the web server."""

import json
import random
import threading
import urllib.error
import urllib.request
from typing import Any, Dict, Iterator, Optional

import pytest

from wonders.play.rating import AGENT_ELO
from wonders.play.server import make_server
from wonders.play.session import (
    HUMAN,
    ConfigError,
    GameConfig,
    HumanSession,
    SeatConfig,
    agent_options,
    legal_action,
    menu_options,
    token_kinds,
)
from wonders.storage import Store


def _config(expansions=('base',), agents=('heuristic', 'board'), human: int = 0, **kwargs: Any) -> GameConfig:
    seats = [SeatConfig(a) for a in agents]
    seats.insert(human, SeatConfig(HUMAN, 'Tester'))
    return GameConfig(expansions=list(expansions), seats=seats, **kwargs)


def _play_out(session: HumanSession, seed: int = 0) -> int:
    rng = random.Random(seed)
    decisions = 0
    while not session.done:
        state = session.state()
        json.dumps(state)  # every state is JSON-ready
        session.act(rng.randrange(len(state['observation']['actions'])))
        decisions += 1
    return decisions


@pytest.mark.parametrize(
    'expansions, agents, human',
    [
        (('base',), ('heuristic', 'board'), 0),
        (('base',), ('random', 'greedy', 'blocker', 'civic', 'board-focus', 'scientist'), 3),
        (('leaders', 'cities'), ('heuristic', 'board', 'random'), 2),
    ],
)
def test_human_plays_a_full_game(expansions, agents, human):
    session = HumanSession(_config(expansions, agents, human, seed=5))
    assert session.human == human
    decisions = _play_out(session)
    assert decisions >= 18  # at least 3 Ages x 6 turns
    state = session.state()
    result = state['result']
    assert len(result['scores']) == len(agents) + 1
    assert 0 <= result['winner'] <= len(agents)
    assert state['players'][human]['name'] == 'Tester'
    record = json.loads(session.record_json())
    assert record['expansions'][0] == 'base'


def test_same_seed_same_game():
    def first_hand(seed: int):
        return [c['name'] for c in HumanSession(_config(seed=seed)).state()['observation']['hand']]

    assert first_hand(11) == first_hand(11)
    assert first_hand(11) != first_hand(12)


def test_human_view_hides_opponents_private_cards():
    session = HumanSession(_config(agents=('hoarder', 'builder'), seed=2))
    for _ in range(4):
        session.act(legal_action(session, 'discard', session.state()['observation']['actions'][0]['card']))
    history = session.state()['observation']['history']
    hidden = [e for e in history if e['seat'] != 0 and e['kind'] in ('wonder', 'discard')]
    assert hidden and all(e['card'] is None for e in hidden)
    assert all(e['card'] is not None for e in history if e['seat'] == 0)


def test_wonder_choice_and_sides():
    session = HumanSession(_config(wonder='The Pyramids of Giza', side='B', opponent_sides='A', seed=1))
    players = session.state()['players']
    assert (players[0]['wonder'], players[0]['side']) == ('The Pyramids of Giza', 'B')
    assert all(p['side'] == 'A' for p in players[1:])
    assert len({p['wonder'] for p in players}) == len(players)


@pytest.mark.parametrize(
    'config, message',
    [
        (GameConfig(seats=[SeatConfig('heuristic')] * 3), 'exactly one human'),
        (GameConfig(seats=[SeatConfig(), SeatConfig(), SeatConfig('board')]), 'exactly one human'),
        (GameConfig(seats=[SeatConfig(), SeatConfig('board')]), '3-7 players'),
        (_config(agents=('heuristic', 'nobody')), 'unknown agent'),
        (_config(expansions=('armada',)), 'not playable yet'),
        (_config(expansions=('atlantis',)), 'unknown expansion'),
        (_config(wonder='Rome'), 'unknown wonder'),
        (_config(side='C'), 'unknown wonder side'),
    ],
)
def test_invalid_configs(config, message):
    with pytest.raises(ConfigError, match=message):
        HumanSession(config)


def test_illegal_action_index():
    session = HumanSession(_config(seed=0))
    with pytest.raises(ConfigError):
        session.act(999)


def test_config_round_trip():
    config = _config(expansions=('leaders',), wonder='The Colosseum of Rome', side='A', seed=7)
    assert GameConfig.from_dict(json.loads(json.dumps(config.to_dict()))) == config
    assert GameConfig.from_dict({'seed': ''}).seed is None


def test_menu_options():
    menu = menu_options()
    ids = {e['id']: e for e in menu['expansions']}
    assert {'base', 'leaders', 'cities', 'armada'} <= set(ids)
    assert ids['armada']['implemented'] is False and ids['armada']['wonders'] == []
    assert 'The Colosseum of Rome' in ids['leaders']['wonders']
    agents = {a['id'] for a in menu['agents']}
    assert {'heuristic', 'board', 'random'} <= agents
    assert (menu['min_players'], menu['max_players']) == (3, 7)
    json.dumps(menu)


def test_neural_opponent_checks_its_vocabulary(tmp_path):
    pytest.importorskip('torch')
    from wonders.env.vocab import vocab_for
    from wonders.ml import save_checkpoint
    from wonders.ml.imitation import new_model

    assert any(a.id == 'neural' for a in agent_options())
    # The bundled model is universal: it plays any expansion combination.
    _play_out(HumanSession(_config(expansions=('leaders', 'cities'), agents=('neural', 'heuristic'), seed=3)))
    base = vocab_for(('base',))
    path = str(tmp_path / 'base-only.pt')
    save_checkpoint(path, new_model(hidden=8, vocab=base)[0], base.card_names, **base.meta())
    HumanSession(_config(agents=('neural:base-only', 'heuristic'), seed=3), checkpoints=[path])
    with pytest.raises(ConfigError, match='not trained'):
        HumanSession(
            _config(expansions=('cities',), agents=('neural:base-only', 'heuristic'), seed=3), checkpoints=[path]
        )


# -- web server ----------------------------------------------------------------
@pytest.fixture
def base_url() -> Iterator[str]:
    httpd = make_server(port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{httpd.server_address[1]}'
    finally:
        httpd.shutdown()
        httpd.server_close()


def _request(url: str, body: Optional[Dict[str, Any]] = None) -> Any:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=60) as res:
        payload = res.read()
        return json.loads(payload) if res.headers['Content-Type'] == 'application/json' else payload.decode()


def test_server_plays_a_game(base_url):
    assert '<title>Wonders Table</title>' in _request(base_url + '/')
    assert _request(base_url + '/favicon.svg').startswith('<svg')
    assert _request(base_url + '/api/state') == {'state': None}
    menu = _request(base_url + '/api/options')
    assert menu['defaults']['seats'][0]['player'] == HUMAN
    state = _request(base_url + '/api/new', {**menu['defaults'], 'seed': 4})['state']
    first = state['observation']['hand'][0]['name']
    assert _request(base_url + '/api/card.svg?name=' + urllib.request.quote(first)).startswith('<svg')
    me = state['players'][0]
    board = f"{base_url}/api/wonder.svg?name={urllib.request.quote(me['wonder'])}&side={me['side']}"
    assert _request(board).startswith('<svg')
    while not state['done']:
        state = _request(base_url + '/api/act', {'index': 0})['state']
    assert state['result']['winner'] in range(3)
    assert _request(base_url + '/api/state')['state']['done'] is True
    record = _request(base_url + '/api/record')
    assert set(record['moves']) == {'1', '2', '3'}


def test_server_reports_errors(base_url):
    with pytest.raises(urllib.error.HTTPError) as err:
        _request(base_url + '/api/act', {'index': 0})
    assert err.value.code == 400 and 'no game' in json.loads(err.value.read())['error']
    with pytest.raises(urllib.error.HTTPError) as err:
        _request(base_url + '/api/new', {'seats': [{'player': 'board'}] * 3})
    assert err.value.code == 400
    with pytest.raises(urllib.error.HTTPError) as err:
        _request(base_url + '/api/card.svg?name=Nope')
    assert err.value.code == 404


def _until_trade_choice(session: HumanSession) -> Dict[str, Any]:
    while not session.done:
        for action in session.state()['observation']['actions']:
            if action.get('payments'):
                return action
        session.act(0)
    pytest.fail('no purchase with a choice of neighbours came up')


def test_human_chooses_whom_to_buy_from():
    from wonders.cards.content import load_content
    from wonders.engine.record import GameRecord

    session = HumanSession(_config(agents=('heuristic', 'board', 'greedy'), human=1, seed=0))
    action = _until_trade_choice(session)
    options = action['payments']
    assert len(options) > 1 and options == sorted(options, key=lambda p: p['total'])
    choice = options[-1]
    with pytest.raises(ConfigError):
        session.act(action['index'], [choice['left'] + 1, choice['right']])
    session.act(action['index'], [choice['left'], choice['right']])
    split = [m for turns in session.env.record().moves.values() for turn in turns for m in turn if len(m) == 4]
    assert [m[1:] for m in split] == [[action['card'], choice['left'], choice['right']]]
    while not session.done:
        session.act(0)
    game = session.env.game
    assert game is not None
    replayed = GameRecord.from_json(session.record_json()).replay(load_content('base'))
    assert [replayed.score(i).total for i in range(4)] == [game.score(i).total for i in range(4)]
    assert [p.coins for p in replayed.players] == [p.coins for p in game.players]


def test_state_describes_cards_and_every_board():
    session = HumanSession(_config(expansions=('base', 'leaders'), seed=4))
    state = session.state()
    assert len(state['boards']) == 3
    for board in state['boards']:
        assert board['stages'] and all('cost' in s and 'effects' in s for s in board['stages'])
    text = session.card_info('Builders Guild')['text']
    assert any('you + neighbours' in line for line in text)
    for name in state['observation']['leaders_hand']:
        assert state['cards'][name]['text']


def test_server_speaks_german(base_url):
    locale = _request(base_url + '/api/locale?lang=de')
    assert locale['_language'] == 'Deutsch' and locale['Start game'] == 'Partie starten'
    assert _request(base_url + '/api/locale') == {}  # English needs no catalog
    menu = _request(base_url + '/api/options?lang=de')
    assert menu['languages']['de'] == 'Deutsch' and menu['lang'] == 'de'
    assert next(e for e in menu['expansions'] if e['id'] == 'base')['name'] == 'Grundspiel'
    assert next(a for a in menu['agents'] if a['id'] == 'greedy')['blurb'] == 'Nimmt sofort die meisten Punkte.'
    assert _request(base_url + '/api/options')['expansions'][0]['name'] == 'Base game'  # cached per language
    state = _request(base_url + '/api/new?lang=de', {**menu['defaults'], 'seed': 4})['state']
    assert state['players'][0]['name'] == 'Du'
    card = state['observation']['hand'][0]['name']  # names stay English ids; the views are translated
    assert state['cards'][card]['text'][0].startswith('Kosten: ')
    me = state['players'][0]
    assert me['wonder_title'] != me['wonder'] and state['boards'][0]['title'] == me['wonder_title']
    assert 'Bibliothek' in _request(base_url + '/api/card.svg?name=Library&lang=de')
    pyramids = f'{base_url}/api/wonder.svg?name=The%20Pyramids%20of%20Giza&side=A&lang=de'
    assert 'STUFE 1' in _request(pyramids)
    assert _request(base_url + '/api/state')['state']['cards'][card]['text'][0].startswith('Cost: ')
    state = _request(base_url + '/api/act?lang=de', {'index': 0})['state']
    assert all(info['text'][0].startswith('Kosten: ') for info in state['cards'].values())


# -- saved games -----------------------------------------------------------------------------
def test_a_saved_game_restores_mid_game_and_finished():
    session = HumanSession(_config(expansions=('leaders',), agents=('heuristic', 'board', 'random'), seed=7))
    for _ in range(8):
        session.act(len(session.state()['observation']['actions']) - 1)
    saved = json.loads(json.dumps(session.saved()))  # survives JSON
    assert saved['finished'] is False and saved['result'] is None and saved['progress']['age'] >= 1
    assert saved['config']['seed'] == session.seed and saved['players'][1]['agent'] == 'heuristic'
    restored = HumanSession.restore(saved)
    assert restored.id == session.id and restored.record_json() == session.record_json()
    assert restored.state()['observation'] == session.state()['observation']
    _play_out(restored)
    saved = json.loads(json.dumps(restored.saved()))
    result = saved['result']
    assert saved['finished'] and saved['finished_at'] and saved['progress'] is None
    assert sorted(result['places'])[0] == 1 and len(result['places']) == 4
    assert result['totals'] == [s['total'] for s in result['scores']]
    game = restored.env.game
    assert game is not None
    assert [sum(c.values()) for c in result['colors']] == [len(p.cards) for p in game.players]
    again = HumanSession.restore(saved)
    assert again.done and again.state()['result'] == restored.state()['result']


def test_a_saved_game_that_no_longer_matches_is_refused():
    session = HumanSession(_config(seed=2))
    session.act(0)
    saved = session.saved()
    saved['log'][0]['moves']['0'] = ['build', 'No Such Card']
    with pytest.raises(ConfigError, match='no longer matches'):
        HumanSession.restore(saved)


@pytest.fixture
def saving_url(tmp_path) -> Iterator[str]:
    httpd = make_server(port=0, store=Store(tmp_path))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{httpd.server_address[1]}'
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_server_saves_rates_and_reopens_games(saving_url):
    assert _request(saving_url + '/api/options')['saving'] is True
    assert _request(saving_url + '/api/games')['games'] == []
    config = {'seats': [{'player': HUMAN}, {'player': 'random'}, {'player': 'hoarder'}], 'seed': 1}
    state = _request(saving_url + '/api/new', config)['state']
    first = state['id']
    state = _request(saving_url + '/api/act', {'index': 0})['state']
    games = _request(saving_url + '/api/games')
    assert [g['id'] for g in games['games']] == [first] and games['games'][0]['current'] is True
    assert games['games'][0]['finished'] is False and 'log' not in games['games'][0]
    assert games['rating']['current'] == games['rating']['start'] and games['rating']['history'] == []

    second = _request(saving_url + '/api/new', config)['state']['id']  # the first game stays saved
    state = _request(saving_url + '/api/games/open', {'id': first})['state']  # ...and resumes
    assert state['id'] == first and not state['done']
    while not state['done']:
        state = _request(saving_url + '/api/act', {'index': 0})['state']
    games = _request(saving_url + '/api/games')
    assert [g['id'] for g in games['games']] == [second, first]  # newest first
    done = games['games'][1]
    assert done['finished'] and done['rating']['before'] == games['rating']['start']
    assert games['rating']['games'] == 1 and games['rating']['current'] == done['rating']['after']
    assert games['rating']['history'][0]['place'] == done['result']['places'][0]
    assert [p['elo'] for p in done['players']] == [None, AGENT_ELO['random'], AGENT_ELO['hoarder']]
    record = _request(saving_url + f'/api/games/record?id={first}')
    assert set(record['moves']) == {'1', '2', '3'}
    viewed = _request(saving_url + '/api/games/open', {'id': first})['state']
    assert viewed['done'] and viewed['result'] == state['result']
    achievements = _request(saving_url + '/api/achievements?lang=de')
    assert achievements['games'] == 1 and len(achievements['achievements']) >= 30
    assert achievements['points'] >= 10 and achievements['max_points'] > achievements['points']
    assert any(a['name'] == 'Mülltaucher' for a in achievements['achievements'])
    assert _request(saving_url + '/api/games/delete', {'id': first}) == {'deleted': first}
    assert _request(saving_url + '/api/state') == {'state': None}  # the open game was deleted
    assert [g['id'] for g in _request(saving_url + '/api/games')['games']] == [second]
    with pytest.raises(urllib.error.HTTPError) as err:
        _request(saving_url + '/api/games/open', {'id': '../etc'})
    assert err.value.code == 400
    with pytest.raises(urllib.error.HTTPError) as err:
        _request(saving_url + '/api/games/record?id=nope')
    assert err.value.code == 404


def test_server_without_a_store_saves_nothing(base_url):
    assert _request(base_url + '/api/options')['saving'] is False
    assert _request(base_url + '/api/games')['games'] == []
    with pytest.raises(urllib.error.HTTPError) as err:
        _request(base_url + '/api/games/open', {'id': 'x'})
    assert 'not saved' in json.loads(err.value.read())['error']


def test_favicon_is_the_project_logo():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    favicon = root / 'wonders' / 'play' / 'static' / 'logo.svg'
    assert favicon.read_bytes() == (root / 'docs' / 'images' / 'logo.svg').read_bytes()


def test_token_counters_follow_the_cards_in_play():
    assert token_kinds(['base', 'leaders']) == {'debt': False, 'diplomacy': False}
    assert token_kinds(['base', 'cities']) == {'debt': True, 'diplomacy': True}
    assert token_kinds(['base', 'wonderpack']) == {'debt': False, 'diplomacy': True}  # a Wonder Pack stage
