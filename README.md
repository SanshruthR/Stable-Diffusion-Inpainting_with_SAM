# Segment Anything & Stable Diffusion Inpainting

![Python](https://img.shields.io/badge/Python-3.7%2B-FF7F50?style=for-the-badge&logo=python)
![License](https://img.shields.io/badge/License-MIT-1E90FF?style=for-the-badge)
![Hugging Face](https://img.shields.io/badge/Powered%20by-Hugging%20Face-FFD700?style=for-the-badge&logo=huggingface)
![Torch](https://img.shields.io/badge/PyTorch-1.0%2B-FF6347?style=for-the-badge&logo=pytorch)
![Gradio](https://img.shields.io/badge/Gradio-2.0%2B-32CD32?style=for-the-badge&logo=gradio)
![download](https://github.com/user-attachments/assets/ae217f10-20ad-47f0-921c-b1570b4c4903)
## Overview
This project utilizes the Segment Anything Model (SAM) and Stable Diffusion Inpainting Pipeline to enable users to segment and inpaint parts of images based on custom prompts. The pipeline is designed to take an input image, generate segmentation masks, and then apply inpainting based on specified prompts, providing an interactive and customizable experience.
Features
1. **Image Segmentation**: Automatically segments objects within an image using the Segment Anything Model.
2. **Inpainting**: Uses the Stable Diffusion Inpainting Pipeline to generate inpainted images based on the segmented masks and user prompts.
3. **Interactive Interface**: Built with Gradio for easy interaction with the model.
4. **Customizable Prompts**: Users can provide custom prompts to guide the inpainting process.

## Architecture
- **Segment Anything Model**: Used to segment objects in an image to create masks for inpainting.
- **Stable Diffusion Inpainting**: Utilizes the pre-trained Stable Diffusion model to generate inpainted images based on user-provided prompts and selected masks.
SAM Model
- The SAM model is initialized with a checkpoint and a specific model type (e.g., `vit_h`), and is used to generate segmentation masks from an image.

## Acknowledgments

- Meta AI for the Segment Anything Model
- Stability AI for Stable Diffusion
- Hugging Face for model hosting and diffusers library

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
