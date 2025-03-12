import numpy as np
import os
import torch
from PIL import Image
from torchvision.utils import save_image

from inference_model import InferenceModel


def pil_to_tensor(images):
    images = np.array(images).astype(np.float32) / 255.0
    images = torch.from_numpy(images.transpose(2, 0, 1))
    return images


model = InferenceModel()

human_img = Image.open("./example/human/00034_00.jpg")
cloth_image = Image.open("./example/cloth/04469_00.jpg")
cloth_desc = ""
output_file_name = "00034_00_04469_00.jpg"

result_image = model(human_img, cloth_image, cloth_desc)
result_image = pil_to_tensor(result_image)

print(result_image)

os.makedirs("result_single", exist_ok=True)
save_image(result_image, os.path.join("result_single", output_file_name))
