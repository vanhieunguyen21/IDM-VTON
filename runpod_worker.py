import base64
import numpy as np
import os
import runpod
import torch
from PIL import Image
from io import BytesIO
from runpod.serverless.utils import rp_upload, rp_cleanup

from inference_model import InferenceModel

torch.cuda.empty_cache()

model = InferenceModel()


def pil_to_tensor(images):
    images = np.array(images).astype(np.float32) / 255.0
    images = torch.from_numpy(images.transpose(2, 0, 1))
    return images


def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


@torch.inference_mode()
def generate_image(job):
    """
    Generate an image from text using your Model
    """
    job_input = job["input"]

    human_image_base64 = job_input['human_image']
    cloth_image_base64 = job_input['cloth_image']

    try:
        # Decode base64 string image
        human_image_data = base64.b64decode(human_image_base64)
        human_image = Image.open(BytesIO(human_image_data))

        cloth_image_data = base64.b64decode(cloth_image_base64)
        cloth_image = Image.open(BytesIO(cloth_image_data))

        cloth_desc = ""

        # Run inference
        result_image = model(human_image, cloth_image, cloth_desc)
        image_base64 = image_to_base64(result_image)

        results = {
            "image": image_base64,
        }
        return results

    except Exception as e:
        return {"error": str(e)}


runpod.serverless.start({"handler": generate_image})
