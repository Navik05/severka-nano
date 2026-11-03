import math

import pytest
import torch

from severka.model import BigramLanguageModel
from severka.train import estimate_loss


def test_estimate_loss_returns_finite_float():
    torch.manual_seed(0)
    model = BigramLanguageModel(10)
    data = torch.randint(0, 10, (200,))

    loss = estimate_loss(model, data, block_size=8, batch_size=4, eval_batches=3)

    assert isinstance(loss, float)
    assert math.isfinite(loss)


def test_estimate_loss_does_not_change_weights():
    torch.manual_seed(0)
    model = BigramLanguageModel(10)
    data = torch.randint(0, 10, (200,))
    before = [p.detach().clone() for p in model.parameters()]

    estimate_loss(model, data, block_size=8, batch_size=4, eval_batches=3)

    assert all(
        torch.equal(old, new)
        for old, new in zip(before, model.parameters(), strict=True)
    )


def test_estimate_loss_restores_mode():
    model = BigramLanguageModel(10)
    data = torch.randint(0, 10, (200,))

    model.eval()
    estimate_loss(model, data, block_size=8, batch_size=4, eval_batches=3)
    assert not model.training

    model.train()
    estimate_loss(model, data, block_size=8, batch_size=4, eval_batches=3)
    assert model.training


def test_estimate_loss_is_reproducible():
    torch.manual_seed(0)
    model = BigramLanguageModel(10)
    data = torch.randint(0, 10, (200,))

    first = estimate_loss(
        model, data, block_size=8, batch_size=4, eval_batches=3, seed=42
    )
    second = estimate_loss(
        model, data, block_size=8, batch_size=4, eval_batches=3, seed=42
    )

    assert first == second


def test_estimate_loss_rejects_zero_batches():
    model = BigramLanguageModel(10)
    data = torch.randint(0, 10, (200,))

    with pytest.raises(ValueError):
        estimate_loss(model, data, block_size=8, batch_size=4, eval_batches=0)