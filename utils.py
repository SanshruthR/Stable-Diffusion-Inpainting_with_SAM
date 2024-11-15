import PIL
import numpy as np
import copy
import cv2
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')

def show_anns(anns, ax=None):
    if len(anns) == 0:
        return
    if ax is None:
        ax = plt.gca()

    sorted_anns = sorted(enumerate(anns), key=(lambda x: x[1]['area']), reverse=True)

    for original_idx, ann in sorted_anns:
        m = ann['segmentation']
        if m.shape != (512, 512):  # Ensure mask is right size
            m = cv2.resize(m.astype(float), (512, 512))
        
        # Create a random color for this mask
        color_mask = np.random.random(3)
        
        # Create the colored mask
        colored_mask = np.zeros((512, 512, 3))
        for i in range(3):
            colored_mask[:,:,i] = color_mask[i]
        
        # Add the mask with transparency
        ax.imshow(np.dstack([colored_mask, m * 0.35]))
        
        # Find contours of the mask
        contours, _ = cv2.findContours((m * 255).astype(np.uint8), 
                                     cv2.RETR_EXTERNAL, 
                                     cv2.CHAIN_APPROX_SIMPLE)
        
        # Add mask number if contours exist
        if contours:
            # Get the largest contour
            cnt = max(contours, key=cv2.contourArea)
            M = cv2.moments(cnt)
            
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Add text with white color and black outline for visibility
                ax.text(cx, cy, str(original_idx), 
                       color='white', 
                       fontsize=16, 
                       ha='center', 
                       va='center', 
                       fontweight='bold',
                       bbox=dict(facecolor='black', 
                                alpha=0.5, 
                                edgecolor='none', 
                                pad=1))


def create_image_grid(original_image, images, names, rows, columns):
    names = copy.copy(names)
    images = copy.copy(images)
    
    # Filter out empty prompts and their corresponding images
    filtered_images = []
    filtered_names = []
    for img, name in zip(images, names):
        if name.strip():
            filtered_images.append(img)
            filtered_names.append(name)
    
    images = filtered_images
    names = filtered_names

    # Add original image
    images.insert(0, original_image)
    names.insert(0, 'Original')

    fig = plt.figure(figsize=(20, 20))
    
    for idx, (img, name) in enumerate(zip(images, names)):
        ax = fig.add_subplot(rows, columns, idx + 1)
        
        if isinstance(img, PIL.Image.Image):
            ax.imshow(img)
        else:
            ax.imshow(img)
            
        ax.set_title(name, fontsize=12, pad=10)
        ax.axis('off')

    plt.tight_layout()
    return fig
