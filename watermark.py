import cv2 as cv
import numpy as np
import util

# Standard JPEG Quantization Table for Luminance
Q_table = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

# 22 mid-frequency indices in zigzag order
mid_freq_indices = [
    (0, 4), (1, 3), (2, 2), (3, 1), (4, 0),
    (5, 0), (4, 1), (3, 2), (2, 3), (1, 4), (0, 5),
    (0, 6), (1, 5), (2, 4), (3, 3), (4, 2), (5, 1), (6, 0),
    (7, 0), (6, 1), (5, 2), (4, 3)
]

def load_rgb_image(image_path):
    img = cv.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    return cv.cvtColor(img, cv.COLOR_BGR2RGB)

def convert_rgb_to_ycbcr(rgb_image):
    return cv.cvtColor(rgb_image, cv.COLOR_RGB2YCrCb)

def apply_chroma_subsampling(ycbcr_image):
    return ycbcr_image

def segment_into_8x8_blocks(image_data):
    h, w, c = image_data.shape
    h = h - h % 8
    w = w - w % 8
    image_data = image_data[:h, :w, :]
    
    blocks = []
    for i in range(0, h, 8):
        row_blocks = []
        for j in range(0, w, 8):
            row_blocks.append(image_data[i:i+8, j:j+8, :])
        blocks.append(row_blocks)
    return np.array(blocks)

def apply_dct(blocks):
    h_b, w_b, h, w, c = blocks.shape
    dct_blocks = np.zeros_like(blocks, dtype=np.float32)
    for i in range(h_b):
        for j in range(w_b):
            for k in range(c):
                block = np.float32(blocks[i, j, :, :, k])
                dct_blocks[i, j, :, :, k] = cv.dct(block)
    return dct_blocks

# FIX 1: Return int32 instead of float32 to prevent truncation bugs when
# calling int() on values like 2.9999998 (which would silently give 2, not 3),
# flipping embedded LSBs during both embedding and extraction.
def quantize_blocks(dct_blocks):
    h_b, w_b, h, w, c = dct_blocks.shape
    quantized_blocks = np.zeros((h_b, w_b, h, w, c), dtype=np.int32)
    for i in range(h_b):
        for j in range(w_b):
            for k in range(c):
                quantized_blocks[i, j, :, :, k] = np.round(
                    dct_blocks[i, j, :, :, k] / Q_table
                ).astype(np.int32)
    return quantized_blocks

def select_mid_frequency_coefficients(quantized_blocks):
    return quantized_blocks

def embed_lsb_watermark(quantized_blocks, watermark_bits):
    h_b, w_b, h, w, c = quantized_blocks.shape
    bit_idx = 0
    total_bits = len(watermark_bits)
    
    watermarked_blocks = np.copy(quantized_blocks)
    for i in range(h_b):
        for j in range(w_b):
            for idx in mid_freq_indices:
                if bit_idx >= total_bits:
                    return watermarked_blocks
                r, c_idx = idx
                coeff = int(watermarked_blocks[i, j, r, c_idx, 0])
                bit = int(watermark_bits[bit_idx])
                if coeff % 2 != bit:
                    if coeff > 0:
                        coeff -= 1
                    elif coeff < 0:
                        coeff += 1
                    else:
                        coeff = 1 if bit == 1 else 0
                watermarked_blocks[i, j, r, c_idx, 0] = coeff
                bit_idx += 1
    if bit_idx < total_bits:
        print(f"Warning: Watermark too large ({total_bits} bits) for image capacity ({bit_idx} bits). Embedded truncated data.")
    return watermarked_blocks

def extract_watermark(quantized_blocks):
    h_b, w_b, h, w, c = quantized_blocks.shape
    extracted_bits = []
    
    # Extract 32 bits for length
    for i in range(h_b):
        for j in range(w_b):
            for idx in mid_freq_indices:
                r, c_idx = idx
                coeff = int(quantized_blocks[i, j, r, c_idx, 0])
                extracted_bits.append(str(coeff % 2))
                if len(extracted_bits) == 32:
                    break
            if len(extracted_bits) == 32:
                break
        if len(extracted_bits) == 32:
            break
            
    if len(extracted_bits) < 32:
        return ""
        
    length_bin = ''.join(extracted_bits[:32])
    length = int(length_bin, 2)
    
    extracted_bits = []
    bit_idx = 0
    
    for i in range(h_b):
        for j in range(w_b):
            for idx in mid_freq_indices:
                if bit_idx < 32:
                    bit_idx += 1
                    continue
                r, c_idx = idx
                coeff = int(quantized_blocks[i, j, r, c_idx, 0])
                extracted_bits.append(str(coeff % 2))
                if len(extracted_bits) == length:
                    return ''.join(extracted_bits)
    return ''.join(extracted_bits)

def apply_zigzag_scan(watermarked_blocks):
    return watermarked_blocks

def run_length_encode(zigzag_data):
    return zigzag_data

def huffman_encode(rle_data):
    return rle_data

# FIX 2: Save as PNG (lossless) instead of JPEG.
# JPEG re-compression applies its own DCT + quantization cycle, overwriting the
# LSBs you carefully embedded. PNG preserves exact pixel values so the
# DCT → quantize round-trip during extraction recovers the original coefficients.
def generate_jpeg_file(huffman_data, output_path):
    h_b, w_b, h, w, c = huffman_data.shape
    img_h, img_w = h_b * 8, w_b * 8
    reconstructed = np.zeros((img_h, img_w, c), dtype=np.uint8)
    
    for i in range(h_b):
        for j in range(w_b):
            for k in range(c):
                dequantized = huffman_data[i, j, :, :, k] * Q_table
                idct_block = cv.idct(np.float32(dequantized))
                reconstructed[i*8:(i+1)*8, j*8:(j+1)*8, k] = np.clip(idct_block, 0, 255)
                
    bgr_img = cv.cvtColor(reconstructed, cv.COLOR_YCrCb2BGR)

    # Force lossless PNG output regardless of the extension the user typed.
    # Saving as JPEG would apply a second lossy compression cycle and destroy
    # the embedded watermark bits.
    if not output_path.lower().endswith('.png'):
        output_path = output_path.rsplit('.', 1)[0] + '_watermarked.png'
        print(f"Note: Output forced to PNG to preserve watermark integrity: {output_path}")

    cv.imwrite(output_path, bgr_img)
    return bgr_img, output_path

def jpeg_watermark_pipeline(rgb_image_path, watermark_path, output_path):
    rgb_image = load_rgb_image(rgb_image_path)
    ycbcr_image = convert_rgb_to_ycbcr(rgb_image)
    subsampled_image = apply_chroma_subsampling(ycbcr_image)
    blocks = segment_into_8x8_blocks(subsampled_image)
    
    dct_blocks = apply_dct(blocks)
    quantized_blocks = quantize_blocks(dct_blocks)
    selected_coeffs = select_mid_frequency_coefficients(quantized_blocks)
    
    try:
        with open(watermark_path, 'rb') as f:
            wm_bytes = f.read()
    except Exception as e:
        print(f"Failed to read watermark file: {e}")
        return
        
    wm_bits = ''.join(format(byte, '08b') for byte in wm_bytes)
    length_bits = format(len(wm_bits), '032b')
    full_watermark_bits = length_bits + wm_bits
    
    watermarked_blocks = embed_lsb_watermark(selected_coeffs, full_watermark_bits)
    
    zigzag_data = apply_zigzag_scan(watermarked_blocks)
    rle_data = run_length_encode(zigzag_data)
    huffman_data = huffman_encode(rle_data)
    
    reconstructed_img, final_output_path = generate_jpeg_file(huffman_data, output_path)
    
    orig_bgr = cv.imread(rgb_image_path)
    orig_bgr = orig_bgr[:reconstructed_img.shape[0], :reconstructed_img.shape[1], :]
    
    mse = util.calculate_mse(orig_bgr, reconstructed_img)
    psnr = util.calculate_psnr(orig_bgr, reconstructed_img)
    print(f"Embedding successful. Saved to {final_output_path}")
    print(f"MSE: {mse:.4f}")
    print(f"PSNR: {psnr:.4f} dB")

def extract_watermark_pipeline(watermarked_image_path, output_wm_path):
    rgb_image = load_rgb_image(watermarked_image_path)
    ycbcr_image = convert_rgb_to_ycbcr(rgb_image)
    subsampled_image = apply_chroma_subsampling(ycbcr_image)
    blocks = segment_into_8x8_blocks(subsampled_image)
    
    dct_blocks = apply_dct(blocks)
    quantized_blocks = quantize_blocks(dct_blocks)
    
    extracted_bits = extract_watermark(quantized_blocks)
    
    if not extracted_bits:
        print("Failed to extract watermark. The image may not contain one.")
        return
        
    byte_array = bytearray()
    for i in range(0, len(extracted_bits), 8):
        byte_segment = extracted_bits[i:i+8]
        if len(byte_segment) == 8:
            byte_array.append(int(byte_segment, 2))
            
    try:
        with open(output_wm_path, 'wb') as f:
            f.write(byte_array)
        print(f"Watermark successfully extracted to {output_wm_path}")
    except Exception as e:
        print(f"Failed to save extracted watermark: {e}")

def main():
    while True:
        print("\n--- JPEG Watermarking Menu ---")
        print("1. Calculate max insertion size")
        print("2. Embed watermark")
        print("3. Extract watermark")
        print("4. Exit")
        choice = input("Enter choice (1/2/3/4): ")
        
        if choice == '1':
            img_path = input("Enter input image path: ")
            img = cv.imread(img_path)
            if img is not None:
                h, w, _ = img.shape
                max_size = util.calculate_max_insertion_size(w, h, 8, 22)
                print(f"Max insertion size: {max_size} bits")
            else:
                print("Image not found.")
        elif choice == '2':
            img_path = input("Enter input image path: ")
            watermark_path = input("Enter path to watermark file (e.g., wm.jpeg): ")
            out_path = input("Enter output image path (will be saved as .png): ")
            try:
                jpeg_watermark_pipeline(img_path, watermark_path, out_path)
            except Exception as e:
                print(f"Error: {e}")
        elif choice == '3':
            img_path = input("Enter watermarked image path: ")
            out_path = input("Enter output path for extracted watermark (e.g., ext_wm.jpeg): ")
            try:
                extract_watermark_pipeline(img_path, out_path)
            except Exception as e:
                print(f"Error: {e}")
        elif choice == '4':
            break
        else:
            print("Invalid choice.")

if __name__ == '__main__':
    main()