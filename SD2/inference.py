import torch
from PIL import Image
from torchvision import transforms
from diffusion_pipe import StableDiffusionDiffImg2ImgPipeline

device = "cuda"

#This is the default model, you can use other fine tuned models as well
pipe = StableDiffusionDiffImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base",
                                                          torch_dtype=torch.float16).to(device)


def preprocess_image(image):
    image = image.convert("RGB")
    # image = transforms.CenterCrop((image.size[1] // 64 * 64, image.size[0] // 64 * 64))(image)
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

with Image.open("/path/to/image1") as imageFile:
    image1 = preprocess_image(imageFile)

with Image.open("/path/to/image2") as imageFile:
    image2 = preprocess_image(imageFile)

with Image.open("/path/to/map1") as mapFile:
    map1 = preprocess_map(mapFile)
    
if image1.shape!=image2.shape:
    raise AssertionError("Both images must be same size", image1.shape, image2.shape)

# increase num_inference_steps to increase denoising strengths

edited_image = pipe(prompt=[""], image=[image1, image2],
     guidance_scale=0,
     num_images_per_prompt=1,
     negative_prompt=[""], map=map, num_inference_steps=3).images[0]


edited_image.save("output.png")

print("Done!")
