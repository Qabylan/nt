import torch
import torchvision.models as models
from model_starter import style_transfer
from telebot import TeleBot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The training takes more time with a larger image size, so a smaller size is chosen for the CPU
imsize = 512 if torch.cuda.is_available() else 128

# We use the VGG19 model as mentioned in the scientific paper.
cnn = models.vgg19(pretrained=True).features.to(device).eval()

TOKEN = "6952770639:AAHCEfBusirUBtrhetFd262_g3jDOMv8pH8"
bot = TeleBot(TOKEN)
img_type = 'style'
content_img_path = None
style_img_path = None

@bot.message_handler(commands=['start'])
def start(message):
    bot.send_message(message.chat.id, 'Отправьте изображение стиля')

def save_photo(photo_type, message):
    file_id = message.photo[-1].file_id
    file_info = bot.get_file(file_id)
    ext = file_info.file_path.split('.')[-1]
    downloaded_file = bot.download_file(file_info.file_path)
    img_path_save = f"{photo_type}.{ext}"
    with open(img_path_save, 'wb') as new_file:
        new_file.write(downloaded_file)
    return img_path_save

@bot.message_handler(content_types=['photo'])
def content_style(message):
    global img_type, content_img_path, style_img_path
    if img_type == 'style':
        style_img_path = save_photo('style', message)
        bot.send_message(message.chat.id, 'Изображение стиля сохранено')
        bot.send_message(message.chat.id, 'Отправьте изображение контента')
        img_type = 'content'
    elif img_type == 'content':
        content_img_path = save_photo('content', message)
        bot.send_message(message.chat.id, 'Изображение контента сохранено')
        bot.send_message(message.chat.id, 'Ожидайте проводится обработка')
        style_transfer(cnn, imsize, device, style_img_path, content_img_path)
        img_type = 'style'
        with open('output.png', 'rb') as f:
            bot.send_photo(message.chat.id, f)
if __name__ == '__main__':
    bot.polling(none_stop=True)