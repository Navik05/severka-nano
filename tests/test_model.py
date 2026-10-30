
import pytest
import torch
from torch.nn import functional as F

from severka.model import BigramLanguageModel


@pytest.fixture(autouse=True)
def _seed():
    """Каждый тест стартует с одного и того же seed"""
    torch.manual_seed(0)

@pytest.fixture
def vocab_size():
    return 10

@pytest.fixture
def model(vocab_size):
    return BigramLanguageModel(vocab_size)

def test_bigram_forward_logits_are_embedding_rows(model):
    """logits — это строки матрицы весов по индексам inputs"""
    inputs = torch.tensor([[1, 1, 2]])
    logits, _ = model(inputs)
    W = model.embedding.weight
    assert torch.equal(logits[0, 0], W[1])
    assert torch.equal(logits[0, 1], W[1])
    assert torch.equal(logits[0, 2], W[2])

def test_bigram_forward_loss_matches_manual_cross_entropy(model, vocab_size):
    """loss совпадает с эталонным cross_entropy (сверка с ручным расчётом)"""
    inputs = torch.tensor([[0, 1, 2]])
    targets = torch.tensor([[1, 2, 3]])
    _, loss = model(inputs, targets)
    expected = F.cross_entropy(
        model.embedding.weight[inputs].reshape(-1, vocab_size),
        targets.reshape(-1),
    )
    assert torch.allclose(loss, expected)

def test_bigram_forward_backward_populates_gradients(model):
    """Обучение работает: градиенты текут и конечны"""
    inputs = torch.randint(0, 10, (2, 3))
    targets = torch.randint(0, 10, (2, 3))
    _, loss = model(inputs, targets)
    loss.backward()
    grad = model.embedding.weight.grad
    assert grad is not None
    assert grad.shape == model.embedding.weight.shape
    assert torch.isfinite(grad).all()

@pytest.mark.parametrize("batch_size,block_size",
                         [(1, 1), (1, 10), (10, 1)])
def test_bigram_forward_edge_shapes_and_inference_mode(
        model,
        vocab_size,
        batch_size,
        block_size
):
    """Граничные размеры + регресс на targets=None"""
    inputs = torch.randint(0, vocab_size, (batch_size, block_size))
    targets = torch.randint(0, vocab_size, (batch_size, block_size))

    # с targets — loss есть
    logits, loss = model(inputs, targets)
    assert logits.shape == (batch_size, block_size, vocab_size)
    assert loss.ndim == 0

    # без targets — loss is None
    logits, loss = model(inputs)
    assert logits.shape == (batch_size, block_size, vocab_size)
    assert loss is None