import torch
from torch import nn

from severka.data import get_batch


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    eval_batches: int,
    seed: int = 1234,
) -> float:
    """Оценивает средний loss на данных без обновления весов.

    Args:
        model: языковая модель.
        data: одномерный тензор токенов (train или val).
        block_size: длина контекста, как при обучении.
        batch_size: число окон в батче.
        eval_batches: по скольким батчам усреднять.
        seed: seed для фиксированных eval-батчей.

    Returns:
        Средний кросс-энтропийный loss по батчам (eval_batches).
    """
    if eval_batches <= 0:
        raise ValueError("eval_batches должен быть > 0")

    was_training = model.training
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    total_loss = 0.0

    try:
        for _ in range(eval_batches):
            inputs, targets = get_batch(
                data,
                block_size,
                batch_size,
                generator = generator
            )
            _, loss = model(inputs, targets)
            total_loss += loss.item()
    finally:
        model.train(was_training)
    return total_loss / eval_batches