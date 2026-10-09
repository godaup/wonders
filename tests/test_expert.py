"""Tests for expert iteration (skipped unless PyTorch is installed)."""

import pytest

torch = pytest.importorskip('torch')

from wonders.ml import load_checkpoint, pretrained_path  # noqa: E402
from wonders.ml.expert import SearchDataset, collect_search, distill, soft_targets  # noqa: E402


def test_soft_targets_follow_the_values():
    probs = soft_targets([3.0, 1.0, 1.0], temperature=1.0)
    assert abs(sum(probs) - 1) < 1e-9 and probs[0] > probs[1] == probs[2]
    assert soft_targets([3.0, 1.0], temperature=0.01)[0] > 0.999  # cold: the search's choice
    assert abs(soft_targets([3.0, 1.0], temperature=1e6)[0] - 0.5) < 1e-3  # hot: uniform


@pytest.fixture(scope='module')
def data():
    return collect_search(
        pretrained_path(), 2, cells=[(('base', 'leaders'), 3)], depth=1, worlds=2, candidates=3, seed=1
    )


def test_collect_records_every_decision_of_the_searching_seat(data, tmp_path):
    assert len(data) >= 2 * 18  # 18 regular turns per game, plus Leaders draft / recruitment
    assert data.targets.shape == data.probs.shape == (len(data), 3)
    assert torch.allclose(data.probs.sum(dim=1), torch.ones(len(data)))
    picked = data.mask.gather(1, data.targets.clamp(min=0))
    assert bool((picked | (data.targets < 0)).all())  # every target is a legal action
    assert int((data.probs > 0).sum(dim=1).max()) > 1  # searched turns spread over candidates
    path = str(tmp_path / 'data.pt')
    data.save(path)
    assert torch.equal(SearchDataset.load(path).targets, data.targets)


def test_distill_moves_the_network_towards_the_search(data):
    model, _names, _meta = load_checkpoint(pretrained_path())
    metrics = distill(model, data, epochs=2, batch_size=32, lr=1e-3, log=None, device='cpu')
    assert 0.0 <= metrics['val_agreement'] <= 1.0 and metrics['train_loss'] > 0


def test_targets_can_be_re_derived_at_another_temperature(data):
    assert data.scores is not None and data.scores.shape == data.targets.shape
    cold = data.with_temperature(0.01)
    hot = data.with_temperature(1e6)
    searched = (data.probs > 0).sum(dim=1) > 1
    assert torch.allclose(cold.probs.sum(dim=1), torch.ones(len(data)))
    best = torch.nan_to_num(data.scores, nan=-1e9).argmax(dim=1)
    assert bool((cold.probs.argmax(dim=1) == best).all())  # cold: the search's choice
    width = (~torch.isnan(data.scores)).sum(dim=1)
    assert torch.allclose(hot.probs[searched].max(dim=1).values, 1.0 / width[searched].float(), atol=1e-3)
    assert torch.equal(cold.probs[~searched], data.probs[~searched])  # unsearched rows stay certain
