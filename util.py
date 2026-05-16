import cv2 as cv
import numpy as np
import math

def calculate_mse(original_image, modified_image):
    """
    Calculates the Mean Squared Error (MSE) between the original image 
    and the watermarked/compressed image.
    
    Formula: MSE = (1 / (M * N)) * sum((original - modified)^2)
    """
    orig = np.array(original_image, dtype=np.float64)
    mod = np.array(modified_image, dtype=np.float64)
    mse = np.mean((orig - mod) ** 2)
    return mse

def calculate_psnr(original_image, modified_image):
    """
    Calculates the Peak Signal-to-Noise Ratio (PSNR) to evaluate 
    the visual quality of the watermarked image.
    
    Formula: PSNR = 10 * log10((MAX_pixel_value^2) / MSE)
    """
    mse = calculate_mse(original_image, modified_image)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 10 * math.log10((max_pixel ** 2) / mse)
    return psnr

def calculate_max_insertion_size(image_width, image_height, block_size=8, mid_freq_coeffs_per_block=22):
    """
    Calculates the maximum watermark capacity (in bits) that can be safely embedded 
    into the host image based on its dimensions and the block segmentation strategy,
    considering only mid-frequency coefficients.
    
    Parameters:
    - image_width: Width of the host image.
    - image_height: Height of the host image.
    - block_size: Size of the segmented blocks (default is 8 for 8x8 DCT blocks).
    - mid_freq_coeffs_per_block: Number of mid-frequency coefficients selected per block.
    
    Returns:
    - Max insertion size in bits.
    """
    num_blocks_x = image_width // block_size
    num_blocks_y = image_height // block_size
    total_blocks = num_blocks_x * num_blocks_y
    max_insertion_size = total_blocks * mid_freq_coeffs_per_block
    return max_insertion_size