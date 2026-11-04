from pathlib import Path

import torch

from severka.data import encode_text, get_batch, load_text, split_text
from severka.model import BigramLanguageModel
from severka.tokenizer import CharTokenizer
from severka.train import estimate_loss

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw.txt"
CHECKPOINT_PATH = ROOT / "checkpoints" / "bigram.pt"

BLOCK_SIZE = 64
BATCH_SIZE = 32
SEED = 333
MAX_STEPS = 5000
EVAL_INTERVAL = 500
EVAL_BATCHES = 20
LEARNING_RATE = 1e-2
SAMPLE_PROMPT = "Раскольников "
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.8


def sample_text(
    model: BigramLanguageModel,
    tokenizer: CharTokenizer,
    prompt: torch.Tensor,
) -> str:
    """Генерирует пример текста и возвращает модель в исходный режим."""
    was_training = model.training
    model.eval()
    try:
        generated = model.generate(prompt, MAX_NEW_TOKENS, BLOCK_SIZE, TEMPERATURE)
    finally:
        model.train(was_training)
    return tokenizer.decode(generated[0].tolist())


def evaluate(
    model: BigramLanguageModel,
    tokenizer: CharTokenizer,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    prompt: torch.Tensor,
    step: int,
) -> tuple[float, float]:
    """Считает loss для train и val и печатает пример генерации."""
    train_loss = estimate_loss(model, train_data, BLOCK_SIZE, BATCH_SIZE, EVAL_BATCHES)
    val_loss = estimate_loss(model, val_data, BLOCK_SIZE, BATCH_SIZE, EVAL_BATCHES)

    print(f"step {step:5d} | train {train_loss:.4f} | val {val_loss:.4f}")
    print(sample_text(model, tokenizer, prompt))
    print("-" * 160)

    return train_loss, val_loss


def main() -> None:
    torch.manual_seed(SEED)

    # Загрузка текста и деление его на train и val
    data = load_text(DATA_PATH)
    train_text, val_text = split_text(data)
    print(f"train = {len(train_text)}, val = {len(val_text)}")

    # Инициализация словаря на train
    tokenizer = CharTokenizer.fit(train_text)
    print(f"vocab = {len(tokenizer)}")

    unknown = sorted(set(val_text) - set(tokenizer.vocab))
    if unknown:
        raise ValueError(f"В val есть символы вне словаря: {unknown!r}")

    # Перевод train и val в токены
    train_data = encode_text(train_text, tokenizer)
    val_data = encode_text(val_text, tokenizer)
    print(f"train токены = {train_data.numel()}, val токены = {val_data.numel()}")

    # Инициализация процесса обучения
    model = BigramLanguageModel(len(tokenizer))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters = {n_params:,}")
    train_generator = torch.Generator().manual_seed(SEED)
    model.train()

    # Подготовка текстовой подсказки
    prompt = torch.tensor(
        [tokenizer.encode(SAMPLE_PROMPT)],
        dtype=torch.long,
    )

    # Проверка до обучения
    print()
    evaluate(model, tokenizer, train_data, val_data, prompt, step=0)

    # Цикл обучения
    for step in range(1, MAX_STEPS + 1):
        inputs, targets = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE, train_generator)
        _, loss = model(inputs, targets)

        # Очищает старые ошибки перед новым шагом обучения
        optimizer.zero_grad(set_to_none=True)
        # Вычисляет, в каких именно весах ошибка в этот раз и как это исправить
        loss.backward()
        # Делает шаг обучения - изменяет веса модели
        optimizer.step()

        if step % EVAL_INTERVAL == 0 or step == MAX_STEPS:
            evaluate(model, tokenizer, train_data, val_data, prompt, step)

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab": tokenizer.vocab,
            "config": {
                "block_size": BLOCK_SIZE,
                "batch_size": BATCH_SIZE,
                "max_steps": MAX_STEPS,
                "learning_rate": LEARNING_RATE,
            },
        },
        CHECKPOINT_PATH,
    )
    print(f"checkpoint сохранён в {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()