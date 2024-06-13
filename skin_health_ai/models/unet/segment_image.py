import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageShow
from torchvision import transforms

from skin_health_ai.models.unet.unet_transposed import load_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def transform_image(image_path):
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)


def predict(model, image):
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        prediction = model(image).detach().cpu()
        prediction = torch.sigmoid(prediction)
        prediction = (prediction > 0.5).float()
    return prediction


def overlay_mask_on_image(image, mask, alpha=0.5):
    mask = (mask * 255).astype(np.uint8)
    mask = Image.fromarray(mask)
    mask = mask.convert("L")

    image = Image.fromarray((image * 255).astype(np.uint8))
    image = image.convert("RGBA")

    overlay = Image.new("RGBA", image.size, (255, 0, 0, 0))
    overlay.paste((255, 0, 0, int(255 * alpha)), (0, 0), mask)

    combined = Image.alpha_composite(image, overlay)
    return combined.convert("RGB")


def segment_and_plot(image_path, model_path):
    model = load_model(model_path)
    model.to(device)

    image = transform_image(image_path)
    prediction = predict(model, image)

    image = image.squeeze().permute(1, 2, 0).numpy()
    prediction = prediction.squeeze().numpy()

    combined_image = overlay_mask_on_image(image, prediction)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(combined_image)
    axes[1].set_title("Segmented Image")
    axes[1].axis("off")

    plt.show()

    ImageShow.show(combined_image)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python segment_image.py <image_path> <model_path>")
        sys.exit(1)
    image_path = sys.argv[1]
    model_path = sys.argv[2]
    segment_and_plot(image_path, model_path)
