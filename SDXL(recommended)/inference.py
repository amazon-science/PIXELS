import torch
from PIL import Image
from torchvision import transforms
from diffusion_pipe import StableDiffusionXLDiffImg2ImgPipeline
from matplotlib import pyplot as plt
import cv2
import numpy as np

torch.cuda.empty_cache()

device = "cuda"

base = StableDiffusionXLDiffImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to(device)

pipe = StableDiffusionXLDiffImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
).to(device)

# generator = torch.Generator(device="cuda").manual_seed(1234)

def preprocess_image(image):
    image = image.convert("RGB")
    image = transforms.Resize((1024, 1024))(image)
    image = transforms.ToTensor()(image)
    image = image * 2 - 1
    image = image.unsqueeze(0).to(device)
    return image


def preprocess_map(map):
    map = map.convert("L")
    map = transforms.CenterCrop((map.size[1] // 64 * 64, map.size[0] // 64 * 64))(map)
    # convert to tensor
    map = transforms.ToTensor()(map)
    map = map.to(device)
    return map

with Image.open("../assets/source_3.jpg") as imageFile:
    image1= preprocess_image(imageFile)

with Image.open("../assets/exemplar_3.jpg") as imageFile:
    image2= preprocess_image(imageFile)

with Image.open("../assets/map_3.jpg") as mapFile:
    map1 = preprocess_map(mapFile)

if image1.shape!=image2.shape:
    raise AssertionError("Both images must be same size", image1.shape, image2.shape)



prompt = [""]
negative_prompt = ["blurry, shadow polaroid photo, scary angry pose"]

# denoising_strength = (1-denoising_start)
edited_images = pipe(prompt=prompt, original_image=[image1, image2], image=[image1, image2], strength=0.1, guidance_scale=17.5,
                        num_images_per_prompt=1,
                        negative_prompt=negative_prompt,
                        map=map1,
                        num_inference_steps=100, denoising_start=0.6).images[0]

# one can use different denoising starts/guidance_scales to create diffrent edits.
edited_images.save("output.png")

print("Done!")