import math
import numpy as np
from PIL import Image

def calculate_max_insertion_size(image_path, bits_per_block=1):
    """
    Calculate the maximum number of bits that can be inserted into the image.
    Assume the image is padded to a multiple of 8x8 blocks, and each block 
    can hold `bits_per_block` bits.
    """
    img = Image.open(image_path)
    W, H = img.size
    
    blocks_avail = ((H + 7) // 8) * ((W + 7) // 8)
    max_bits = blocks_avail * bits_per_block
    return max_bits

def calculate_mse_psnr(original_path, watermarked_path):
    """
    Calculate the Mean Squared Error (MSE) and Peak Signal-to-Noise Ratio (PSNR)
    between the original image and the watermarked image.
    """
    img1 = np.array(Image.open(original_path).convert('RGB'), dtype=np.float64)
    img2 = np.array(Image.open(watermarked_path).convert('RGB'), dtype=np.float64)
    
    # If the images have different shapes due to padding differences, match them
    if img1.shape != img2.shape:
        min_h = min(img1.shape[0], img2.shape[0])
        min_w = min(img1.shape[1], img2.shape[1])
        img1 = img1[:min_h, :min_w]
        img2 = img2[:min_h, :min_w]

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 10 * math.log10((255.0 ** 2) / mse)
        
    return mse, psnr

def recompress_attack(input_path, output_path, quality):
    """
    Perform a recompression attack by opening the watermarked image
    and saving it again with a specified JPEG quality.
    """
    img = Image.open(input_path)
    img.save(output_path, "JPEG", quality=quality, subsampling=0)
    print(f"Recompression attack completed. Saved as '{output_path}' with Quality={quality}.")

def calculate_ber(original_wm_path, recovered_wm_path):
    """
    Calculate the total bit errors and the Bit Error Rate (BER) between the 
    original watermark and the recovered watermark. Both should be binary images.
    """
    img1 = Image.open(original_wm_path).convert('1')
    img2 = Image.open(recovered_wm_path).convert('1')
    
    # Ensure they are the same size
    if img1.size != img2.size:
        print(f"Warning: Watermark sizes do not match! {img1.size} vs {img2.size}")
        min_w = min(img1.size[0], img2.size[0])
        min_h = min(img1.size[1], img2.size[1])
        img1 = img1.crop((0, 0, min_w, min_h))
        img2 = img2.crop((0, 0, min_w, min_h))

    arr1 = np.array(img1, dtype=bool)
    arr2 = np.array(img2, dtype=bool)

    total_bits = arr1.size
    bit_errors = int(np.sum(arr1 != arr2))
    ber = bit_errors / total_bits if total_bits > 0 else 0.0

    return bit_errors, ber, total_bits
