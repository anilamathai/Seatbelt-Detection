import torch
import argparse
import cv2
import numpy as np
from torchvision import transforms

from model import get_model
from utils import load_config, load_checkpoint

#explicit mapping from model output indices to labels:::::
CLASS_MAP = {
    0: "No-Seatbelt",
    1: "Seatbelt",
    2: "VLQ"
}

VLQ_THRESHOLD = 0.4

def preprocess(image_path, image_size):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    image = transform(image)
    return image.unsqueeze(0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    args = parser.parse_args()

    config = load_config("config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Load model::::
    model = get_model(num_classes=3, pretrained=False).to(device)
    model = load_checkpoint(
        model,
        f"{config['paths']['checkpoint_dir']}/best_model.pth",
        device
    )
    model.eval()

    #preprocess image:::::
    image = preprocess(args.image, config["training"]["image_size"]).to(device)

    #forward pass::::
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    pred_label = CLASS_MAP[pred_idx]

    if pred_label == "VLQ":
        confidence = 0.0
    else:
        confidence = float(probs[pred_idx])

    if probs[2] > VLQ_THRESHOLD or confidence < config["inference"]["confidence_threshold"]:
        confidence = 0.0

    print({
        "prediction": pred_label,
        "confidence": round(confidence, 6)
    })

if __name__ == "__main__":
    main()