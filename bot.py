import asyncio
import logging
import os
from typing import Set, Tuple
import aiohttp
from aiogram import Bot, Dispatcher, F
from aiogram.enums import ChatAction
from aiogram.filters import CommandStart
from aiogram.types import Message

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s"
)

QUESTION_TEXT = (
    "Сколько нужно сгенерировать постов "
    "(от 1 до 10, значительно дешевле 1 раз 10 постов чем 10 раз по 1 посту)?"
)

# Очередь моделей: если первая перегружена или недоступна, переходит к следующей
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash"
]


def load_config(env_path: str = "tokens.env") -> Tuple[str, str, Set[int]]:
    """
    Загружает токены и разрешенные ID из tokens.env с разделителем ';'
    или из системных переменных окружения (GitHub Secrets / ENV).
    """
    raw_data = {}
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or ";" not in line:
                    continue
                key, val = line.split(";", 1)
                raw_data[key.strip()] = val.strip()

    ai_token = raw_data.get("AI_TOKEN") or os.getenv("AI_TOKEN")
    tg_token = raw_data.get("TG_TOKEN") or os.getenv("TG_TOKEN")
    ids_raw = raw_data.get("IDS") or os.getenv("IDS", "")

    if not ai_token or not tg_token:
        raise ValueError(
            "Не найдены AI_TOKEN или TG_TOKEN! "
            "Укажите их в tokens.env или через переменные окружения (GitHub Secrets)."
        )

    allowed_ids: Set[int] = set()
    for item in ids_raw.split(","):
        cleaned = item.strip()
        if cleaned.isdigit():
            allowed_ids.add(int(cleaned))

    return ai_token, tg_token, allowed_ids


def load_dataset(dataset_path: str = "dataset.txt") -> str:
    """Загружает текст из dataset.txt с сохранением форматирования."""
    if not os.path.exists(dataset_path):
        logging.warning(f"Файл {dataset_path} не найден! Будет передан пустой датасет.")
        return ""
    with open(dataset_path, "r", encoding="utf-8") as f:
        return f.read()


async def request_gemini(prompt: str, ai_token: str) -> str:
    """
    Последовательно опрашивает модели из списка GEMINI_MODELS.
    При 503, 429 или других сбоях автоматически переключается дальше по цепочке.
    """
    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ]
    }

    async with aiohttp.ClientSession() as session:
        for model in GEMINI_MODELS:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={ai_token}"
            logging.info(f"Отправка запроса к модели {model}...")

            try:
                timeout = aiohttp.ClientTimeout(total=120)
                async with session.post(url, json=payload, timeout=timeout) as response:
                    if response.status == 200:
                        data = await response.json()
                        candidates = data.get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            text = "".join(part.get("text", "") for part in parts)
                            if text.strip():
                                logging.info(f"Успешный ответ от {model}")
                                return text
                    else:
                        err_body = await response.text()
                        logging.warning(
                            f"Модель {model} вернула HTTP {response.status}: {err_body}"
                        )
            except Exception as e:
                logging.warning(f"Ошибка при обращении к {model}: {e}")

    models_str = ", ".join(GEMINI_MODELS)
    raise RuntimeError(f"Не удалось получить ответ ни от одной из моделей: {models_str}.")


async def send_chunked_message(message: Message, text: str, max_length: int = 4000) -> None:
    """Безопасная отправка длинных сообщений без превышения лимита Telegram (4096 символов)."""
    for i in range(0, len(text), max_length):
        await message.answer(text[i:i + max_length])


async def main():
    # Загрузка конфигурации
    ai_token, tg_token, allowed_ids = load_config("tokens.env")
    logging.info(f"Загружено разрешенных ID: {len(allowed_ids)}")

    bot = Bot(token=tg_token)
    dp = Dispatcher()

    @dp.message(CommandStart())
    async def handle_start(message: Message):
        user_id = message.from_user.id if message.from_user else None
        if user_id not in allowed_ids:
            await message.answer("Access Denied")
            return

        await message.answer(QUESTION_TEXT)

    @dp.message(F.text)
    async def handle_text(message: Message):
        user_id = message.from_user.id if message.from_user else None

        # Проверка безопасности
        if user_id not in allowed_ids:
            await message.answer("Access Denied")
            return

        user_input = message.text.strip()

        # Валидация введенного значения
        try:
            n = int(user_input)
        except ValueError:
            error_reason = f"Значение «{user_input}» не является целым числом."
            await message.answer(f"{error_reason}\nПожалуйста, введите целое число от 1 до 10.\n\n{QUESTION_TEXT}")
            return

        if n < 1 or n > 10:
            error_reason = f"Число {n} вне допустимого диапазона (должно быть от 1 до 10 включительно)."
            await message.answer(f"{error_reason}\nПожалуйста, попробуйте снова.\n\n{QUESTION_TEXT}")
            return

        # Индикация работы бота
        await bot.send_chat_action(chat_id=message.chat.id, action=ChatAction.TYPING)
        status_msg = await message.answer(f"Генерирую {n} постов через Gemini, пожалуйста, подождите...")

        # Загрузка датасета и сборка промпта
        dataset_content = load_dataset("dataset.txt")
        prompt = (
            "Роль: Автор уютного русскоязычного Telegram-канала «Interesting English».\n"
            f"Выведи ТОЛЬКО готовый текст {n} постов на основе предыдущих "
            "(стиль, формат, логическая цепочка), старайся делать разнообразные. "
            "Между постами обязательно вставляй строку <next> для парсинга ботом. "
            "Никаких вступительных или заключительных слов, не используй Markdown и форматирование там, где он не применялся в предыдущих постах.\n"
            "Предыдущие посты (от самого первого до последнего):\n"
            f"{dataset_content}"
        )

        try:
            raw_response = await request_gemini(prompt=prompt, ai_token=ai_token)
        except Exception as e:
            logging.error(f"Ошибка генерации: {e}")
            await status_msg.edit_text("Произошла ошибка при обращении к нейросети. Попробуйте позже.")
            await message.answer(QUESTION_TEXT)
            return

        # Парсинг постов по маркеру <next>
        posts = [p.strip() for p in raw_response.split("<next>") if p.strip()]

        # Удаляем временное статусное сообщение перед выводом постов
        try:
            await status_msg.delete()
        except Exception:
            pass

        if not posts:
            await message.answer("Нейросеть вернула пустой ответ. Попробуйте еще раз.")
        else:
            for idx, post in enumerate(posts, 1):
                post_content = f"ПОСТ №{idx}\n{post}"
                await send_chunked_message(message, post_content)

        # Возврат в начало цикла
        await message.answer(QUESTION_TEXT)

    # Обработка любых других типов сообщений (стикеры, фото, голосовые)
    @dp.message()
    async def handle_other_messages(message: Message):
        user_id = message.from_user.id if message.from_user else None
        if user_id not in allowed_ids:
            await message.answer("Access Denied")
            return
        await message.answer(f"Ожидается текстовое число от 1 до 10.\n\n{QUESTION_TEXT}")

    logging.info("Бот запущен и ожидает сообщений...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
