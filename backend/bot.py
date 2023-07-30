import logging
import os
import time
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
logging.info(f"Loading model and tokenizer. MODEL_NAME: {MODEL_NAME}, MODEL_FILE_NAME: {MODEL_FILE_NAME}.")
MODEL, TOKENIZER = toxicity_checker.load_model_tokenizer(
    MODEL_NAME,
    f"./toxicity_checker/data/{MODEL_FILE_NAME}",
)
logging.info(f"Loading model and tokenizer to {next(iter(MODEL.parameters())).device} done.")

# Initialize bot and dispatcher
bot = Bot(token=API_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)


@dp.message_handler(state="*", commands=["start", "help"])
async def send_welcome(message: types.Message):
    await message.reply(emojize("""*Welcome to Toxicity Checker Bot!*:robot:\n
:biohazard:I'll checking for toxicity every message sent to me,
so you can add me to your group for toxicity control.:biohazard:\n
All messages with toxicity percent more than acceptable 
in a chat will be recognized as toxic and informing message will be send.
By default, acceptable toxicity is 70%, but you can change 
it by command `/set_acceptable_toxicity_percent <positive integer between 0 and 100>`.
In groups this command available only for administrators. 
To view current acceptable toxicity percent in a chat, use `/acceptable_toxicity_percent` command.\n
Also you can use command `/toxicity <text>` for checking 
following text, or just reply some message and write 
`/toxicity` without arguments to check the replayed one.\n"""),
                        parse_mode=types.ParseMode.MARKDOWN)


@dp.message_handler(state="*", commands=["set_acceptable_toxicity_percent"])
async def check_toxicity_command(message: types.Message):
    new_acceptable_toxicity_percent = message.get_args()
    if (new_acceptable_toxicity_percent
            and 0 <= int(new_acceptable_toxicity_percent) <= 100):
        new_acceptable_toxicity_percent = int(new_acceptable_toxicity_percent)
    else:
        new_acceptable_toxicity_percent = 70
    state = dp.current_state(chat=message.chat.id)
    current_acceptable_toxicity_percent = await state.get_state()
    if current_acceptable_toxicity_percent is None:
        current_acceptable_toxicity_percent = 70
    sender_id = message["from"].id
    sender_username = message["from"].username
    if message.chat.type == "private":
        reply_message = (f"Acceptable toxicity reset from {current_acceptable_toxicity_percent}%"
                         f" to {new_acceptable_toxicity_percent}% by @{sender_username}")
        await state.set_state(new_acceptable_toxicity_percent)
    else:
        administrators = await message.chat.get_administrators()
        for administrator in administrators:
            if administrator.user.id == sender_id:
                reply_message = (f"Acceptable toxicity reset from {current_acceptable_toxicity_percent}%"
                                 f" to {new_acceptable_toxicity_percent}% by @{sender_username}")
                await state.set_state(new_acceptable_toxicity_percent)
                break
        else:
            reply_message = f"Insufficient permissions"
    await message.reply(reply_message)


@dp.message_handler(state="*", commands=["acceptable_toxicity_percent"])
async def check_toxicity_command(message: types.Message):
    #logging.info(message)
    state = dp.current_state(chat=message.chat.id)
    current_acceptable_toxicity_percent = await state.get_state()
    if current_acceptable_toxicity_percent is None:
        current_acceptable_toxicity_percent = 70
    current_acceptable_toxicity_percent = int(current_acceptable_toxicity_percent)
    #logging.info(current_acceptable_toxicity_percent)
    reply_message = f"Acceptable toxicity is {current_acceptable_toxicity_percent}%"
    await message.reply(reply_message)


@dp.message_handler(Text(startswith=["/toxicity"]), state="*")
async def check_toxicity_command(message: types.Message):
    #logging.info(message)
    #time_start = time.time()
    if message.text == "/toxicity":
        message = message.reply_to_message if hasattr(message, "reply_to_message") else None
        if message is None:
            await message.reply("Wrong `/toxicity` command usage!")

    text_author = message["from"].username
    text = message.text
    if text[:9] == "/toxicity":
        text = text[9:].strip()

    state = dp.current_state(chat=message.chat.id)
    current_acceptable_toxicity_percent = await state.get_state()
    if current_acceptable_toxicity_percent is None:
        current_acceptable_toxicity_percent = 70
    current_acceptable_toxicity_percent = int(current_acceptable_toxicity_percent)
    threshold = current_acceptable_toxicity_percent / 100

    text_is_toxic, toxicity_score = toxicity_checker.check_toxicity(text, MODEL, TOKENIZER, threshold)
    reply_message = f"Message '{text}'\nis toxic on {toxicity_score * 100:.0f}%"
    # reply_message = (f"WowWowWow, more respect please, @{text_author}! Your message '{text}'"
    #                  f" toxic on {toxicity_score * 100:.0f}%")
    # if not text_is_toxic:
    #     reply_message = (f"Nice to hear that, @{text_author}! Your message '{text}'"
    #                      f" toxic on {toxicity_score * 100:.0f}%")
    #logging.info(f"Message '{message}' processed in {time.time() - time_start:.3f}s")
    await message.reply(reply_message)


@dp.message_handler(state="*")
async def check_toxicity(message: types.Message):
    #logging.info(message)
    #time_start = time.time()
    text_author = message["from"].username
    text = message.text

    state = dp.current_state(chat=message.chat.id)
    current_acceptable_toxicity_percent = await state.get_state()
    if current_acceptable_toxicity_percent is None:
        current_acceptable_toxicity_percent = 70
    current_acceptable_toxicity_percent = int(current_acceptable_toxicity_percent)
    threshold = current_acceptable_toxicity_percent / 100

    text_is_toxic, toxicity_score = toxicity_checker.check_toxicity(text, MODEL, TOKENIZER, threshold)
    reply_message = f":biohazard:Toxic message from @{text_author}! Toxicity: {toxicity_score * 100:.0f}%"
    # reply_message = (f"WowWowWow, more respect please, @{text_author}! Your message '{text}'"
    #                  f" toxic on {toxicity_score * 100:.0f}%")
    if not text_is_toxic:
        if TRIGGER_ON == "toxic":
            return
        reply_message = (f"Nice to hear that, @{text_author}! Your message '{text}'"
                         f" toxic on {toxicity_score * 100:.0f}%")
    #logging.info(f"Message '{message}' processed in {time.time() - time_start:.3f}s")
    await message.reply(emojize(reply_message))


async def shutdown(dispatcher: Dispatcher):
    await dispatcher.storage.close()
    await dispatcher.storage.wait_closed()


if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True, on_shutdown=shutdown)
