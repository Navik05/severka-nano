
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

def test_generate_shape_and_prompt(model):
    torch.manual_seed(0)
    prompt = torch.tensor([[1, 2, 3]])
    out = model.generate(prompt, max_new_tokens=5, block_size=4)
    assert out.shape == (1, 8)
    assert torch.equal(out[:, :3], prompt)
    assert out.min() >= 0
    assert out.max() < 10

def test_generate_zero_temperature_is_deterministic(model):
    prompt = torch.tensor([[1, 2, 3]])
    a = model.generate(prompt, 5, 4, temperature=0.0)
    b = model.generate(prompt, 5, 4, temperature=0.0)
    assert torch.equal(a, b)

def test_generate_sampling_reproducible_with_seed(model):
    prompt = torch.tensor([[1, 2, 3]])
    torch.manual_seed(42)
    a = model.generate(prompt, 5, 4, temperature=1.0)
    torch.manual_seed(42)
    b = model.generate(prompt, 5, 4, temperature=1.0)
    assert torch.equal(a, b)

def test_generate_zero_new_tokens(model):
    prompt = torch.tensor([[1, 2, 3]])
    assert torch.equal(model.generate(prompt, 0, 4), prompt)

@pytest.mark.parametrize("block_size", [0, -1])
def test_generate_rejects_bad_block_size(model, block_size):
    with pytest.raises(ValueError):
        model.generate(torch.tensor([[1, 2, 3]]), 1, block_size)

def test_generate_rejects_empty_prompt(model):
    with pytest.raises(ValueError):
        model.generate(torch.zeros((1, 0), dtype=torch.long), 1, 4)

def test_generate_batch(model):
    prompt = torch.tensor([[1, 2], [3, 4]])
    out = model.generate(prompt, 5, 4)
    assert out.shape == (2, 7)