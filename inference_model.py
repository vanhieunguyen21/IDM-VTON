from PIL import Image
import gradio as gr
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)
from diffusers import DDPMScheduler, AutoencoderKL
from typing import List

import torch
import os
from transformers import AutoTokenizer
import numpy as np
from utils_mask import get_mask_location
from torchvision import transforms
import apply_net
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import save_image


class InferenceModel:
    def __init__(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load models
        base_path = 'yisol/IDM-VTON'
        example_path = os.path.join(os.path.dirname(__file__), 'example')

        unet = UNet2DConditionModel.from_pretrained(
            base_path,
            subfolder="unet",
            torch_dtype=torch.float16,
        )
        unet.requires_grad_(False)
        tokenizer_one = AutoTokenizer.from_pretrained(
            base_path,
            subfolder="tokenizer",
            revision=None,
            use_fast=False,
        )
        tokenizer_two = AutoTokenizer.from_pretrained(
            base_path,
            subfolder="tokenizer_2",
            revision=None,
            use_fast=False,
        )
        noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")

        text_encoder_one = CLIPTextModel.from_pretrained(
            base_path,
            subfolder="text_encoder",
            torch_dtype=torch.float16,
        )
        text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
            base_path,
            subfolder="text_encoder_2",
            torch_dtype=torch.float16,
        )
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            base_path,
            subfolder="image_encoder",
            torch_dtype=torch.float16,
        )
        vae = AutoencoderKL.from_pretrained(
            base_path,
            subfolder="vae",
            torch_dtype=torch.float16,
        )

        # "stabilityai/stable-diffusion-xl-base-1.0",
        UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
            base_path,
            subfolder="unet_encoder",
            torch_dtype=torch.float16,
        )

        parsing_model = Parsing(0)
        openpose_model = OpenPose(0)

        UNet_Encoder.requires_grad_(False)
        image_encoder.requires_grad_(False)
        vae.requires_grad_(False)
        unet.requires_grad_(False)
        text_encoder_one.requires_grad_(False)
        text_encoder_two.requires_grad_(False)
        tensor_transfrom = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        pipe = TryonPipeline.from_pretrained(
            base_path,
            unet=unet,
            vae=vae,
            feature_extractor=CLIPImageProcessor(),
            text_encoder=text_encoder_one,
            text_encoder_2=text_encoder_two,
            tokenizer=tokenizer_one,
            tokenizer_2=tokenizer_two,
            scheduler=noise_scheduler,
            image_encoder=image_encoder,
            torch_dtype=torch.float16,
        )
        pipe.unet_encoder = UNet_Encoder

        self.device = device
        self.parsing_model = parsing_model
        self.openpose_model = openpose_model
        self.tensor_transform = tensor_transfrom
        self.pipe = pipe

    @staticmethod
    def pil_to_binary_mask(pil_image, threshold=0):
        np_image = np.array(pil_image)
        grayscale_image = Image.fromarray(np_image).convert("L")
        binary_mask = np.array(grayscale_image) > threshold
        mask = np.zeros(binary_mask.shape, dtype=np.uint8)
        for i in range(binary_mask.shape[0]):
            for j in range(binary_mask.shape[1]):
                if binary_mask[i, j] == True:
                    mask[i, j] = 1
        mask = (mask * 255).astype(np.uint8)
        output_mask = Image.fromarray(mask)
        return output_mask

    @staticmethod
    def pil_to_tensor(images):
        images = np.array(images).astype(np.float32) / 255.0
        images = torch.from_numpy(images.transpose(2, 0, 1))
        return images

    @torch.inference_mode()
    def __call__(self, human_img: Image, cloth_img: Image, cloth_desc: str, denoise_steps=30, seed=42):
        device = self.device
        openpose_model = self.openpose_model
        parsing_model = self.parsing_model
        tensor_transform = self.tensor_transform
        pipe = self.pipe

        openpose_model.preprocessor.body_estimation.model.to(device)
        pipe.to(device)
        pipe.unet_encoder.to(device)

        human_img = human_img.convert("RGB").resize((768, 1024))
        cloth_img = cloth_img.convert("RGB").resize((768, 1024))

        # Generate mask
        keypoints = openpose_model(human_img.resize((384, 512)))
        model_parse, _ = parsing_model(human_img.resize((384, 512)))
        mask, mask_gray = get_mask_location('hd', "upper_body", model_parse, keypoints)
        mask = mask.resize((768, 1024))
        # mask_gray = (1 - transforms.ToTensor()(mask)) * tensor_transform(human_img)
        # mask_gray = to_pil_image((mask_gray + 1.0) / 2.0)

        # Generate dense pose
        human_img_arg = _apply_exif_orientation(human_img.resize((384, 512)))
        human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")

        args = apply_net.create_argument_parser().parse_args(('show', './configs/densepose_rcnn_R_50_FPN_s1x.yaml',
                                                              './ckpt/densepose/model_final_162be9.pkl', 'dp_segm',
                                                              '-v',
                                                              '--opts', 'MODEL.DEVICE', 'cuda'))
        # verbosity = getattr(args, "verbosity", None)
        pose_img = args.func(args, human_img_arg)
        pose_img = pose_img[:, :, ::-1]
        pose_img = Image.fromarray(pose_img).resize((768, 1024))

        # Embed for human image
        prompt = "model is wearing " + cloth_desc
        negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
        with torch.inference_mode():
            (prompt_embeds,
             negative_prompt_embeds,
             pooled_prompt_embeds,
             negative_pooled_prompt_embeds,
             ) = pipe.encode_prompt(
                prompt,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )

        # Embed for cloth image
        prompt = "a photo of " + cloth_desc
        negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
        if not isinstance(prompt, List):
            prompt = [prompt] * 1
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * 1
        with torch.inference_mode():
            (prompt_embeds_c, _, _, _,) = pipe.encode_prompt(
                prompt,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
                negative_prompt=negative_prompt,
            )

        pose_img = tensor_transform(pose_img).unsqueeze(0).to(device, torch.float16)
        cloth_tensor = tensor_transform(cloth_img).unsqueeze(0).to(device, torch.float16)
        generator = torch.Generator(device).manual_seed(seed) if seed is not None else None
        images = pipe(
            prompt_embeds=prompt_embeds.to(device, torch.float16),
            negative_prompt_embeds=negative_prompt_embeds.to(device, torch.float16),
            pooled_prompt_embeds=pooled_prompt_embeds.to(device, torch.float16),
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device, torch.float16),
            num_inference_steps=denoise_steps,
            generator=generator,
            strength=1.0,
            pose_img=pose_img.to(device, torch.float16),
            text_embeds_cloth=prompt_embeds_c.to(device, torch.float16),
            cloth=cloth_tensor.to(device, torch.float16),
            mask_image=mask,
            image=human_img,
            height=1024,
            width=768,
            ip_adapter_image=cloth_img.resize((768, 1024)),
            guidance_scale=2.0,
        )[0]

        return images[0]
