'''
HW8 BI_Python - Telegram Bot for upscaling images.
Project team members:
     - Kikalova Tatiana
     - Жожиков Леонид
     - Муроцмев Антон
     - Куприянов Семён
'''
import logging

from aiogram import Bot, types
from aiogram.utils import executor
from aiogram.utils.emoji import emojize
from aiogram.dispatcher import Dispatcher
from aiogram.types.message import ContentType
from aiogram.utils.markdown import text, bold
from aiogram.types import ParseMode
import shutil
import cv2
from config import TOKEN
from cnn_model import classify, load_trained_resnet18

# enable logging
logging.basicConfig(format=u'%(filename)s [ LINE:%(lineno)+3s ]'
                           u'#%(levelname)+8s [%(asctime)s]  %(message)s',
                    level=logging.INFO)

# global objects
bot = Bot(token=TOKEN)
dp = Dispatcher(bot)


DATA_PATH = '.'
model = load_trained_resnet18(epoch=21)

@dp.message_handler(commands=['start'])
async def process_start_command(message: types.Message):
    await message.reply('Hi!\nI am Bird Classifier Bot. Type /help '
                        'to know how to use me.')


@dp.message_handler(commands=['help'])
async def process_help_command(message: types.Message):
    msg = text('Send me an image as a document to classify it.'
               'And I will text you back a name of a bird')
    await message.reply(msg, parse_mode=ParseMode.MARKDOWN)


@dp.message_handler(content_types=['document'])
async def process_document_message(msg: types.Message):

    # check that document is image
    if msg.document.mime_type[0:6] != 'image/':
        msg_text = text(emojize('Bad document. :neutral_face:'),
                        'This is not an image.')
        await msg.reply(msg_text, parse_mode=ParseMode.MARKDOWN)
        return

    # make pahts
    user_dir = 'files/id' + str(msg.from_user.id) + '/'
    img_path_orig = user_dir + msg.document.file_name

    # download image
    await msg.document.download(destination_file=img_path_orig)

    # classify image
    try:
        res = classify(model, DATA_PATH, img_path_orig)
    except BaseException as ex:
        logging.error(f'process_photo_message() failed: {ex}')
        msg_text = text(emojize('I can\'t classify this image. '
                                'Something went wrong. '
                                ':face_with_spiral_eyes:'))
        await msg.reply(msg_text, parse_mode=ParseMode.MARKDOWN)
        return

    reply_text = '\n'.join(res)
    logging.info(f'''{img_path_orig}:\n{reply_text}''')
    # send reply
    await msg.reply(reply_text, parse_mode=ParseMode.MARKDOWN)


@dp.message_handler(content_types=['photo'])
async def process_photo_message(msg: types.Message):

    # make pahts
    user_dir = 'files/id' + str(msg.from_user.id) + '/'

    # prcoess only one photo, ideally should be: for photo in msg.photo:
    img_path_orig = user_dir + msg.photo[0].file_unique_id
    await msg.photo[0].download(destination_file=img_path_orig)

    # classify image
    try:
        res = classify(model, DATA_PATH, img_path_orig)
    except BaseException as ex:
        logging.error(f'process_photo_message() failed: {ex}')
        msg_text = text(emojize('I can\'t classify this image. '
                                'Something went wrong. '
                                ':face_with_spiral_eyes:'))
        await msg.reply(msg_text, parse_mode=ParseMode.MARKDOWN)
        return

    reply_text = text(emojize('You sent me a compressed photo :nerd_face:,'),
                      emojize('classification gonna be bad :stuck_out_tongue_winking_eye:\n'),
                      'Better send me a',
                      bold('uncompressed document'),
                      '\n')
    reply_text += '\n'.join(res)
    logging.info(f'''{img_path_orig}:\n{reply_text}''')
    # send reply
    await msg.reply(reply_text, parse_mode=ParseMode.MARKDOWN)


@dp.message_handler(content_types=ContentType.ANY)
async def unknown_message(msg: types.Message):

    message_text = text('I don\'t know what to do with this ',
                        emojize(':astonished:'),
                        '\nUse /help please.')
    await msg.reply(message_text, parse_mode=ParseMode.MARKDOWN)


if __name__ == '__main__':

    # start bot
    executor.start_polling(dp)
