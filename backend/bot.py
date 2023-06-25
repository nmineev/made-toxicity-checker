import logging
import os
import sys
import pprint
from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.utils import markdown
from aiogram.dispatcher.filters import Text
from emoji import emojize
import random
random.seed(123)
import toxicity_checker


# Configure logging
logging.basicConfig(level=logging.INFO)

# Set Global Variables
#sys.path.append(os.path.join(os.path.dirname(__file__), "toxicity_checker"))
API_TOKEN = os.environ["TOKEN"]
MODEL_NAME = os.environ["MODEL_NAME"]
MODEL_FILE_NAME = os.environ["MODEL_FILE_NAME"]
TRIGGER_ON = "toxic"  # Available values: "all", "toxic"

# Load model and tokenizer
MODEL, TOKENIZER = toxicity_checker.load_model_tokenizer(
    MODEL_NAME,
    f"./toxicity_checker/data/{MODEL_FILE_NAME}",
)

# Initialize bot and dispatcher
bot = Bot(token=API_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)


@dp.message_handler(commands=["start", "help"])
async def send_welcome(message: types.Message):
    await message.reply("""*Welcome to Toxicity Checker Bot!*\n
Just send me your message and i check it for toxicity.""",
                        parse_mode=types.ParseMode.MARKDOWN)


@dp.message_handler(Text(startswith=["/toxicity"]))
async def check_toxicity_command(message: types.Message):
    #logging.info(message)
    if message.text == "/toxicity":
        message = message.reply_to_message if hasattr(message, "reply_to_message") else None
        if message is None:
            await message.reply("Wrong `/toxicity` command usage!")

    text_author = message["from"].username
    text = message.text
    if text[:9] == "/toxicity":
        text = text[9:].strip()

    text_is_toxic, toxicity_score = toxicity_checker.check_toxicity(text, MODEL, TOKENIZER)
    reply_message = (f"WowWowWow, more respect please, @{text_author}! Your message '{text}'"
                     f" toxic on {toxicity_score * 100:.0f}%")
    if not text_is_toxic:
        reply_message = (f"Nice to hear that, @{text_author}! Your message '{text}'"
                         f" toxic on {toxicity_score * 100:.0f}%")
    await message.reply(reply_message)


@dp.message_handler()
async def check_toxicity(message: types.Message):
    #logging.info(message)
    # if message.text == "/toxicity":
    #     message = message.reply_to_message if hasattr(message, "reply_to_message") else None
    #     if message is None:
    #         await message.reply("Wrong `/toxicity` command usage!")
    # #logging.info(message)

    text_author = message["from"].username
    text = message.text
    # if text[:9] == "/toxicity":
    #     text = text[9:].strip()

    text_is_toxic, toxicity_score = toxicity_checker.check_toxicity(text, MODEL, TOKENIZER)
    reply_message = (f"WowWowWow, more respect please, @{text_author}! Your message '{text}'"
                     f" toxic on {toxicity_score * 100:.0f}%")
    if not text_is_toxic:
        if TRIGGER_ON == "toxic":
            return
        reply_message = (f"Nice to hear that, @{text_author}! Your message '{text}'"
                         f" toxic on {toxicity_score * 100:.0f}%")
    await message.reply(reply_message)

#@dp.message_handler(commands=["toxicity"])



if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)
