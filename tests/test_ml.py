"""Tests for the learned agent (skipped unless PyTorch is installed)."""

import random

import pytest

torch = pytest.importorskip('torch')

from click.testing import CliRunner  # noqa: E402

from wonders import cli  # noqa: E402
from wonders._mp import get_context  # noqa: E402
from wonders.agents import GreedyAgent, RandomAgent  # noqa: E402
from wonders.agents.bench import AGENTS, AgentSpec, run_bench  # noqa: E402
from wonders.agents.match import play_match  # noqa: E402
from wonders.cards.content import load_content  # noqa: E402
from wonders.engine.record import PlayerSetup  # noqa: E402
from wonders.env.vocab import vocab_for  # noqa: E402
from wonders.ml import NeuralAgent, PolicyValueNet, load_checkpoint, save_checkpoint  # noqa: E402
from wonders.ml.imitation import collect, new_model, train_imitation  # noqa: E402
from wonders.ml.ppo import PPOConfig, _gae, train_ppo  # noqa: E402

SETUPS = [
    PlayerSetup('N', 'The Pyramids of Giza', 'A'),
    PlayerSetup('G', 'The Colossus of Rhodes', 'B'),
    PlayerSetup('R', 'The Hanging Gardens of Babylon', 'B'),
]


@pytest.fixture(scope='module')
def trained(tmp_path_factory):
    """A tiny model cloned from a few heuristic games, saved to disk."""
    torch.manual_seed(0)
    vocab = vocab_for(('base',))
    data = collect(8, seed=0, vocab=vocab)
    model, names = new_model(hidden=32, vocab=vocab)
    metrics = train_imitation(model, data, epochs=2, batch_size=64, log=None)
    path = str(tmp_path_factory.mktemp('ckpt') / 'tiny.pt')
    save_checkpoint(path, model, names, source='test', **vocab.meta())
    return model, names, path, data, metrics


def test_policy_masks_illegal_actions():
    net = PolicyValueNet(obs_size=5, n_actions=4, hidden=8)
    mask = torch.tensor([[True, False, True, False]])
    logits, value = net(torch.randn(1, 5), mask)
    assert value.shape == (1,)
    probs = torch.softmax(logits, dim=1)[0]
    assert probs[1] == 0 and probs[3] == 0 and float(probs.sum()) == pytest.approx(1.0)


def test_collect_and_imitate(trained):
    model, names, _path, data, metrics = trained
    assert len(data) > 100
    assert data.obs.shape == (len(data), model.config['obs_size'])
    assert data.mask.shape == (len(data), len(names) * 4)
    assert bool(data.mask[torch.arange(len(data)), data.action].all())  # teacher actions were legal
    assert 0.0 < metrics['val_accuracy'] <= 1.0


def test_checkpoint_round_trip(trained):
    model, names, path, data, _metrics = trained
    loaded, loaded_names, meta = load_checkpoint(path)
    assert loaded_names == names and meta['source'] == 'test'
    x, m = data.obs[:5], data.mask[:5]
    with torch.no_grad():
        assert torch.equal(model(x, m)[0], loaded(x, m)[0])


def test_neural_agent_plays_legal_games(trained):
    _model, _names, path, _data, _metrics = trained
    content = load_content('base')
    agents = [NeuralAgent(path), GreedyAgent(), RandomAgent(random.Random(0))]
    game, record = play_match(content, SETUPS, agents, rng=random.Random(1))
    replayed = record.replay(content)
    assert [replayed.score(i).total for i in range(3)] == [game.score(i).total for i in range(3)]
    sampler_a = NeuralAgent(path, sample=True, rng=random.Random(5))
    sampler_b = NeuralAgent(path, sample=True, rng=random.Random(5))
    game_a, _ = play_match(content, SETUPS, [sampler_a, GreedyAgent(), GreedyAgent()], rng=random.Random(2))
    game_b, _ = play_match(content, SETUPS, [sampler_b, GreedyAgent(), GreedyAgent()], rng=random.Random(2))
    assert game_a.score(0).as_dict() == game_b.score(0).as_dict()


def test_neural_agent_rejects_mismatched_vocab(trained):
    model, names, _path, _data, _metrics = trained
    with pytest.raises(ValueError):
        NeuralAgent(model=model, card_names=names[:-1])
    with pytest.raises(ValueError):
        NeuralAgent()


def test_bench_runs_neural_agent_in_worker_processes(trained):
    # torch is loaded here, so pools must not fork (inherited OpenMP state deadlocks).
    assert get_context().get_start_method() in ('forkserver', 'spawn')
    _model, _names, path, _data, _metrics = trained
    spec = AgentSpec.of('neural', NeuralAgent, checkpoint=path)
    serial = run_bench([spec, AGENTS['greedy'], AGENTS['random']], n_games=4, seed=1)
    parallel = run_bench([spec, AGENTS['greedy'], AGENTS['random']], n_games=4, seed=1, processes=2)
    assert parallel.table() == serial.table()


def test_gae_matches_hand_computation():
    rewards = [torch.tensor([1.0]), torch.tensor([2.0])]
    values = [torch.tensor([0.5]), torch.tensor([0.25])]
    dones = [torch.tensor([0.0]), torch.tensor([1.0])]
    adv = _gae(rewards, values, dones, torch.tensor([9.0]), gamma=1.0, lam=0.5)
    # t=1 ends the episode: delta1 = 2 - 0.25; t=0: delta0 = 1 + 0.25 - 0.5, adv0 = delta0 + 0.5 * adv1
    assert adv[1, 0] == pytest.approx(1.75)
    assert adv[0, 0] == pytest.approx(0.75 + 0.5 * 1.75)


def test_ppo_runs_and_reports(trained):
    model, _names, _path, _data, _metrics = trained
    config = PPOConfig(n_envs=2, envs_per_worker=0, rollout_steps=24, iterations=2, minibatch=16)
    before = [p.clone() for p in model.parameters()]
    history = train_ppo(model, config, log=None)
    assert len(history) == 2 and {'win_rate', 'margin', 'approx_kl', 'games'} <= set(history[-1])
    assert history[0]['games'] + history[1]['games'] >= 2  # 48 learner steps per env > 2 games
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))


def test_cli_train_and_play_neural(tmp_path):
    out = tmp_path / 'cli.pt'
    runner = CliRunner()
    trained = runner.invoke(
        cli.main, ['train', 'imitate', '-n', '4', '--epochs', '1', '--hidden', '16', '-o', str(out)]
    )
    assert trained.exit_code == 0, trained.output
    played = runner.invoke(cli.main, ['play', '-q', '-a', f'neural:{out},greedy,random'])
    assert played.exit_code == 0, played.output
    assert 'neural:cli' in played.output
    tuned = runner.invoke(
        cli.main,
        [
            'train',
            'ppo',
            '--init',
            str(out),
            '-n',
            '1',
            '--envs',
            '2',
            '--envs-per-worker',
            '0',
            '-o',
            str(tmp_path / 'p.pt'),
        ],
    )
    assert tuned.exit_code == 0, tuned.output
    assert (tmp_path / 'p.pt').exists()


def test_bundled_pretrained_agent_plays():
    from wonders.ml import pretrained_path

    content = load_content('base')
    agents = [NeuralAgent(pretrained_path()), GreedyAgent(), RandomAgent(random.Random(0))]
    game, _record = play_match(content, SETUPS, agents, rng=random.Random(3))
    assert 0 <= game.winner() < 3
    played = CliRunner().invoke(cli.main, ['play', '-q', '-a', 'neural,greedy,random'])
    assert played.exit_code == 0, played.output
    assert 'neural:ppo_universal_v2' in played.output
    # The bundled model is universal: every expansion combination and player count.
    played = CliRunner().invoke(
        cli.main,
        ['play', '-q', '-e', 'base', '-e', 'leaders', '-e', 'cities', '-a', 'neural,greedy,heuristic,random,board'],
    )
    assert played.exit_code == 0, played.output


def test_checkpoints_record_their_encoder(tmp_path):
    vocab = vocab_for(('base',))
    model, names = new_model(hidden=8, vocab=vocab)
    with pytest.raises(ValueError, match='vocab.meta'):  # a v2 model without meta could never be loaded
        save_checkpoint(str(tmp_path / 'x.pt'), model, names)
    legacy = vocab.with_encoder('card-v1')
    old_model, _ = new_model(hidden=8, vocab=legacy)
    save_checkpoint(str(tmp_path / 'old.pt'), old_model, names)  # legacy shape: loads as card-v1
    assert NeuralAgent(str(tmp_path / 'old.pt')).vocab.encoder_kind == 'card-v1'
    save_checkpoint(str(tmp_path / 'new.pt'), model, names, **vocab.meta())
    assert NeuralAgent(str(tmp_path / 'new.pt')).vocab.encoder_kind == 'table-v2'


def test_cli_lineup_allows_repeated_agents_with_neural(trained, tmp_path):
    _model, _names, path, _data, _metrics = trained
    runner = CliRunner()
    ok = runner.invoke(cli.main, ['bench', '-n', '3', '-a', f'neural:{path},neural:{path},heuristic,heuristic'])
    assert ok.exit_code == 0, ok.output
    other = tmp_path / 'tiny.pt'  # same file name as the fixture's checkpoint, different file
    with open(path, 'rb') as fh:
        other.write_bytes(fh.read())
    clash = runner.invoke(cli.main, ['bench', '-n', '1', '-a', f'neural:{path},neural:{other},greedy'])
    assert clash.exit_code != 0
