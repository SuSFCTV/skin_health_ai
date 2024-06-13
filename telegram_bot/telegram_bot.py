import logging
import torch
from PIL import Image
from io import BytesIO
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
from torchvision import transforms

from skin_health_ai.models.unet.segment_image import predict, overlay_mask_on_image
from skin_health_ai.models.unet.unet_transposed import load_model

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = load_model("../models/unet_model.pth")
model.to(device)


def transform_image(image: Image.Image):
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )
    return transform(image).unsqueeze(0)


async def start(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text("Привет! Отправьте мне изображение, и я выполню его сегментацию.")


async def handle_image(update: Update, context: CallbackContext) -> None:
    try:
        logger.info("Получено изображение")
        photo = update.message.photo[-1]
        photo_file = await photo.get_file()
        photo_bytes = await photo_file.download_as_bytearray()
        image = Image.open(BytesIO(photo_bytes))
        logger.info("Изображение успешно загружено и преобразовано")

        transformed_image = transform_image(image)
        prediction = predict(model, transformed_image)
        image_np = transformed_image.squeeze().permute(1, 2, 0).numpy()
        prediction_np = prediction.squeeze().numpy()

        combined_image = overlay_mask_on_image(image_np, prediction_np)
        bio = BytesIO()
        bio.name = "result.png"
        combined_image.save(bio, "PNG")
        bio.seek(0)
        await update.message.reply_photo(photo=bio)
        logger.info("Изображение успешно отправлено пользователю")
    except Exception as e:
        logger.error(f"Ошибка при обработке изображения: {e}")
        await update.message.reply_text("Произошла ошибка при обработке изображения.")


async def error(update: Update, context: CallbackContext) -> None:
    logger.warning('Update "%s" caused error "%s"', update, context.error)


def main():
    application = Application.builder().token("TOKEN").build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO, handle_image))
    application.add_error_handler(error)
    application.run_polling()


if __name__ == "__main__":
    main()
