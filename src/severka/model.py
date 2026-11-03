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

    @torch.no_grad()
    def generate(
            self,
            inputs: torch.Tensor,
            max_new_tokens: int,
            block_size: int,
            temperature: float = 1.0,
    ) -> torch.Tensor:
        """Достраивает последовательность inputs на max_new_tokens токенов.

        Args:
            inputs: тензор индексов токенов формы (batch_size, sequence_length),
                    где sequence_length - текущая длина последовательности.
            max_new_tokens: количество новых токенов, которое необходимо сгенерировать и добавить к inputs.
            block_size: максимальный размер контекста, который модель может обработать за раз.
            temperature: коэффициент масштабирования логитов (logits) регулирующий случайность генерации.
                    =0 - жадный выбор, всегда самый вероятный токен; temperature > 0 - сэмплирование.
        Returns:
            torch.Tensor: тензор индексов токенов формы (batch_size, sequence_length + max_new_tokens),
                          содержащий исходную последовательность вместе со сгенерированными токенами.

        """ # noqa: E501
        if inputs.ndim != 2:
            raise ValueError("inputs должен иметь форму (batch_size, sequence_length)")
        if inputs.size(1) == 0:
            raise ValueError("inputs не должен быть пустым: нужен хотя бы один токен")
        if temperature < 0:
            raise ValueError("temperature должна быть >= 0")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens должен быть >= 0")
        if block_size <= 0:
            raise ValueError("block_size должен быть > 0")

        for _ in range(max_new_tokens):
            # Вырезаем последний контекст, который модель способна вместить
            inputs_block = inputs[:, -block_size:]
            # Получаем предсказания модели
            logits, _ = self(inputs_block)
            # Берем логиты самого последнего токена
            logits = logits[:, -1, :]

            # Выбираем следующий токен в зависимости от температуры
            if temperature == 0:
                # Жадный выбор (самый вероятный токен)
                inputs_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                # Температурное сэмплирование
                probabilities = F.softmax(logits / temperature, dim=-1)
                inputs_next = torch.multinomial(probabilities, num_samples=1)

            # Приклеиваем токен к общей истории
            inputs = torch.cat((inputs, inputs_next), dim=1)

        return inputs