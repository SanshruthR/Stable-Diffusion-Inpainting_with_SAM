import torch
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image, to_tensor
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from diffusers import StableDiffusionInpaintPipeline, EulerDiscreteScheduler
from PIL import Image
import numpy as np
import cv2

class SegmentAnythingModel:
    def __init__(self, sam_checkpoint, model_type, device):
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=device)
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=32,
            pred_iou_thresh=0.99,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100
        )
        self.target_size = (512, 512)

    def preprocess_image(self, image):
        """Resize image to 512x512"""
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Get current dimensions
        width, height = image.size
        
        # Resize to 512x512 directly
        image = image.resize(self.target_size, Image.Resampling.LANCZOS)
        return np.array(image)

    def generate_masks(self, image):
        processed_image = self.preprocess_image(image)
        return self.mask_generator.generate(processed_image)

class StableDiffusionInpaintingPipeline:
    def __init__(self, model_dir):
        # Initialize the scheduler first
        self.scheduler = EulerDiscreteScheduler.from_pretrained(model_dir, subfolder="scheduler")
        
        # Initialize the pipeline with the scheduler
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_dir,
            scheduler=self.scheduler,
            revision="fp16",
            torch_dtype=torch.float16
        )
        self.pipe = self.pipe.to("cuda")
        self.pipe.enable_xformers_memory_efficient_attention()
        self.target_size = (512, 512)

    def preprocess_image(self, image):
        """Ensure image is in the right format and size"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return image.resize(self.target_size, Image.Resampling.LANCZOS)

    def inpaint(self, prompt, image, mask_image, guidance_scale=10, num_inference_steps=60, generator=None):
        """
        Args:
            prompt (str): The prompt for inpainting
            image (PIL.Image or np.ndarray): The original image
            mask_image (PIL.Image or np.ndarray): The mask for inpainting
            guidance_scale (float): Higher guidance scale encourages images that are closer to the prompt
            num_inference_steps (int): Number of denoising steps
            generator (torch.Generator): Generator for reproducibility
        """
        # Preprocess images
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if isinstance(mask_image, np.ndarray):
            mask_image = Image.fromarray(mask_image)

        # Resize images
        image = image.resize(self.target_size, Image.Resampling.LANCZOS)
        mask_image = mask_image.resize(self.target_size, Image.Resampling.NEAREST)

        # Run inpainting
        output = self.pipe(
            prompt=prompt,
            image=image,
            mask_image=mask_image,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            height=512,
            width=512
        )
        
        return output.images[0]
