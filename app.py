import gradio as gr
import torch
from PIL import Image
import numpy as np
from engine import SegmentAnythingModel, StableDiffusionInpaintingPipeline
from utils import show_anns, create_image_grid
import matplotlib.pyplot as plt
import PIL
import requests
import matplotlib
matplotlib.use('Agg')  # Use Agg backend

# Download SAM checkpoint
url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
response = requests.get(url)

with open("sam_vit_h_4b8939.pth", "wb") as file:
    file.write(response.content)

# Initialize models
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
sam_model = SegmentAnythingModel(sam_checkpoint, model_type, device)

model_dir = "stabilityai/stable-diffusion-2-inpainting"
sd_pipeline = StableDiffusionInpaintingPipeline(model_dir)

# Global variable to store masks
current_masks = None
current_image = None

def segment_image(image):
    global current_masks, current_image
    current_image = image
    
    # Convert to numpy array
    image_array = np.array(image)
    
    # Generate masks
    current_masks = sam_model.generate_masks(image_array)
    
    # Create visualization of masks
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    
    # Display the original image first
    ax.imshow(sam_model.preprocess_image(image))
    
    # Overlay masks
    show_anns(current_masks, ax)
    
    ax.axis('off')
    plt.tight_layout()
    
    return fig

def inpaint_image(mask_index, prompt1, prompt2, prompt3, prompt4):
    global current_masks, current_image
    
    if current_masks is None or current_image is None:
        return None
    
    # Get selected mask
    segmentation_mask = current_masks[mask_index]['segmentation']
    stable_diffusion_mask = PIL.Image.fromarray((segmentation_mask * 255).astype(np.uint8))

    # Generate inpainted images
    prompts = [p for p in [prompt1, prompt2, prompt3, prompt4] if p.strip()]
    generator = torch.Generator(device="cuda").manual_seed(42)  # Fixed seed for consistency
    
    encoded_images = []
    for prompt in prompts:
        img = sd_pipeline.inpaint(
            prompt=prompt,
            image=Image.fromarray(np.array(current_image)),
            mask_image=stable_diffusion_mask,
            guidance_scale=7.5,  # Lower guidance scale for more creative results
            num_inference_steps=50,  # Good balance between quality and speed
            generator=generator
        )
        encoded_images.append(img)

    # Create result grid
    result_grid = create_image_grid(Image.fromarray(np.array(current_image)),
                                  encoded_images,
                                  prompts,
                                  2, 3)
    
    return result_grid

# Create Gradio interface with two tabs
with gr.Blocks() as demo:
    gr.Markdown("# Segment Anything + Stable Diffusion Inpainting")
    
    with gr.Tab("Step 1: Segment Image"):
        with gr.Row():
            input_image = gr.Image(label="Input Image")
            mask_output = gr.Plot(label="Available Masks")
        segment_btn = gr.Button("Generate Masks")
        segment_btn.click(fn=segment_image, inputs=[input_image], outputs=[mask_output])
    
    with gr.Tab("Step 2: Inpaint"):
        with gr.Row():
            with gr.Column():
                mask_index = gr.Slider(minimum=0, maximum=20, step=1, 
                                     label="Mask Index (select based on mask numbers from Step 1)")
                prompt1 = gr.Textbox(label="Prompt 1", placeholder="Enter first inpainting prompt")
                prompt2 = gr.Textbox(label="Prompt 2", placeholder="Enter second inpainting prompt")
                prompt3 = gr.Textbox(label="Prompt 3", placeholder="Enter third inpainting prompt")
                prompt4 = gr.Textbox(label="Prompt 4", placeholder="Enter fourth inpainting prompt")
            inpaint_output = gr.Plot(label="Inpainting Results")
        inpaint_btn = gr.Button("Generate Inpainting")
        inpaint_btn.click(fn=inpaint_image, 
                         inputs=[mask_index, prompt1, prompt2, prompt3, prompt4],
                         outputs=[inpaint_output])

if __name__ == "__main__":
    demo.launch(share=True, debug=True)
