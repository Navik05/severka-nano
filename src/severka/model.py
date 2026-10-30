import torch
import torch.nn as nn
from torch.nn import functional as F


class BigramLanguageModel(nn.Module):
    """Биграммная языковая модель: предсказывает следующий токен по текущему

     Модель хранит обучаемую матрицу переходов формы (vocab_size, vocab_size):
     строка i — это логиты следующего токена после токена i. Контекст длиннее одного токена не учитывается.
    """ # noqa: E501

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(
            self,
            inputs: torch.Tensor,
            targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Вычисляет логиты и кросс-энтропийную потерю.

        Args:
            inputs: индексы токенов формы (batch_size, block_size) из get_batch, значения в диапазоне [0, vocab_size)
            targets: правильные следующие токены той же формы, что и ``inputs``. Если None, модель работает в режиме инференса
            и ``loss`` равен None.
        Returns:
            кортеж ``(logits, loss)``

            - ``logits`` - тензор формы (batch_size, block_size, vocab_size), собранный как выборка строк из
              матрицы весов self.embedding(vocab_size, vocab_size) по индексам из inputs(batch_size, block_size).
              Каждая строка - это vocab_size сырых оценок того, насколько вероятен каждый следующий токен.
              Порядок строк в logits совпадает с порядком токенов в inputs.
            - ``loss`` - 0-мерный тензор, средняя кросс-энтропия - насколько модель в среднем ошибается,
              предсказывая следующий токен.
        """ # noqa: E501
        logits = self.embedding(inputs)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
        return logits, loss