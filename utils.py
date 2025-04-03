from PIL import Image, ImageDraw, ImageFont
import torch
import re

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    
def extract_action(output_decode: str) -> int:
    pattern = r"<action>\s*([012])\s*</action>"
    matches = re.findall(pattern, output_decode, flags=re.IGNORECASE)
    if not matches:
        raise ValueError("No valid <action> tag found in the output.")
    return int(matches[-1])

def ascii_to_image(ascii_text: str, num_cols=80, num_rows=24) -> Image.Image:
    font = ImageFont.load_default()
    bbox = font.getbbox("A")
    char_width = bbox[2] - bbox[0]
    char_height = bbox[3] - bbox[1]
    img_width, img_height = num_cols * char_width, num_rows * char_height
    image = Image.new("RGB", (img_width, img_height), "black")
    draw = ImageDraw.Draw(image)
    lines = ascii_text.splitlines()
    for row, line in enumerate(lines[:num_rows]):
        draw.text((0, row * char_height), line.ljust(num_cols), font=font, fill="white")
    return image