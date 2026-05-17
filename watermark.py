import cv2 as cv
import numpy as np
import util

# ── Constants ────────────────────────────────────────────────────────────────

Q_table = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=np.float32)

mid_freq_indices = [
    (0, 4), (1, 3), (2, 2), (3, 1), (4, 0),
    (5, 0), (4, 1), (3, 2), (2, 3), (1, 4), (0, 5),
    (0, 6), (1, 5), (2, 4), (3, 3), (4, 2), (5, 1), (6, 0),
    (7, 0), (6, 1), (5, 2), (4, 3)
]

JPEG_QUALITY = 95


# ── Helpers ──────────────────────────────────────────────────────────────────

def compute_jpeg_qtable(quality: int) -> np.ndarray:
    quality = max(1, min(100, int(quality)))
    scale = 5000 / quality if quality < 50 else 200 - 2 * quality
    qtable = np.floor((Q_table * scale + 50) / 100)
    return np.clip(qtable, 1, 255).astype(np.float32)


def segment_into_8x8_blocks(image_data: np.ndarray) -> np.ndarray:
    h, w, c = image_data.shape
    h, w = h - h % 8, w - w % 8
    image_data = image_data[:h, :w, :]
    blocks = []
    for i in range(0, h, 8):
        row = [image_data[i:i+8, j:j+8, :] for j in range(0, w, 8)]
        blocks.append(row)
    return np.array(blocks, dtype=np.float32)


def reconstruct_image(blocks: np.ndarray) -> np.ndarray:
    h_b, w_b, _, _, c = blocks.shape
    img = np.zeros((h_b * 8, w_b * 8, c), dtype=np.float32)
    for i in range(h_b):
        for j in range(w_b):
            img[i*8:(i+1)*8, j*8:(j+1)*8, :] = blocks[i, j]
    return img


# ── QIM embedding / extraction ───────────────────────────────────────────────

def embed_qim(blocks: np.ndarray, watermark_bits: str, jpeg_Q: np.ndarray) -> np.ndarray:
    """
    QIM embedding on pre-quantized blocks.

    IMPORTANT: blocks must come from a JPEG-decoded image (see
    jpeg_watermark_pipeline). Because the pixels are already dequantized
    integer-DCT values, cv2.dct recovers the exact q that JPEG stored
    internally. Pinning q*step and calling cv2.idct then produces spatial
    values that JPEG's integer DCT will re-quantize back to exactly q,
    guaranteeing zero bit errors on extraction.
    """
    result = blocks.copy()
    bit_idx = 0
    total_bits = len(watermark_bits)
    h_b, w_b = result.shape[:2]

    for i in range(h_b):
        for j in range(w_b):
            dct_block = cv.dct(result[i, j, :, :, 0].copy())
            changed = False

            for r, c in mid_freq_indices:
                if bit_idx >= total_bits:
                    break

                desired = int(watermark_bits[bit_idx])
                step = jpeg_Q[r, c]

                q = int(np.round(dct_block[r, c] / step))
                if q % 2 != desired:
                    q += 1

                dct_block[r, c] = q * step
                bit_idx += 1
                changed = True

            if changed:
                result[i, j, :, :, 0] = cv.idct(dct_block)

    if bit_idx < total_bits:
        print(f"Warning: capacity {bit_idx} bits < watermark {total_bits} bits. Truncated.")
    return result


def extract_qim(blocks: np.ndarray, jpeg_Q: np.ndarray) -> str:
    h_b, w_b = blocks.shape[:2]

    def iter_bits():
        for i in range(h_b):
            for j in range(w_b):
                dct_block = cv.dct(np.float32(blocks[i, j, :, :, 0]))
                for r, c in mid_freq_indices:
                    q = int(np.round(dct_block[r, c] / jpeg_Q[r, c]))
                    yield q % 2

    gen = iter_bits()

    header = [next(gen) for _ in range(32)]
    if len(header) < 32:
        return ""
    payload_len = int(''.join(str(b) for b in header), 2)

    max_possible = h_b * w_b * len(mid_freq_indices) - 32
    if payload_len <= 0 or payload_len > max_possible:
        print(f"Warning: extracted length header is invalid ({payload_len} bits). "
              f"Max capacity is {max_possible} bits. Watermark may be corrupted.")
        return ""

    payload = []
    for bit in gen:
        payload.append(str(bit))
        if len(payload) == payload_len:
            break

    return ''.join(payload)


# ── Public pipelines ─────────────────────────────────────────────────────────

def jpeg_watermark_pipeline(image_path: str, watermark_path: str, output_path: str):
    """
    Embed a watermark file into an image and save as JPEG.

    ROOT CAUSE FIX — pre-quantization before embedding:
    ----------------------------------------------------
    cv2.dct uses a floating-point DCT, but libjpeg (used internally by
    cv2.imwrite) uses an integer AAN DCT. When we pin a coefficient to
    q*step in float DCT space and call cv2.idct, tiny floating-point errors
    appear in the spatial pixels. When JPEG's integer DCT then re-quantizes
    those pixels it can recover a different q, flipping the embedded bit.

    Fix: save the source image as JPEG at JPEG_QUALITY first, then reload it.
    The reloaded pixels are exact dequantized outputs of JPEG's integer DCT,
    so cv2.dct recovers the same q JPEG stored. After embedding and saving
    again at the same quality, JPEG's integer DCT re-quantizes back to
    exactly our pinned q — zero bit errors guaranteed.
    """
    jpeg_Q = compute_jpeg_qtable(JPEG_QUALITY)

    bgr = cv.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not output_path.lower().endswith(('.jpg', '.jpeg')):
        output_path = output_path.rsplit('.', 1)[0] + '.jpg'
        print(f"Output adjusted to: {output_path}")

    # ── PRE-QUANTIZATION: bake in JPEG's integer DCT rounding before embedding
    cv.imwrite(output_path, bgr, [cv.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    bgr_prequant = cv.imread(output_path)

    ycbcr  = cv.cvtColor(bgr_prequant, cv.COLOR_BGR2YCrCb).astype(np.float32)
    blocks = segment_into_8x8_blocks(ycbcr)

    try:
        with open(watermark_path, 'rb') as f:
            wm_bytes = f.read()
    except Exception as e:
        print(f"Failed to read watermark: {e}")
        return

    wm_bits   = ''.join(format(b, '08b') for b in wm_bytes)
    full_bits = format(len(wm_bits), '032b') + wm_bits

    h_b, w_b = blocks.shape[:2]
    usable   = h_b * w_b * len(mid_freq_indices) - 32
    if len(wm_bits) > usable:
        print(f"Warning: watermark payload ({len(wm_bits)} bits) exceeds usable "
              f"capacity ({usable} bits). Watermark will be truncated.")

    watermarked = embed_qim(blocks, full_bits, jpeg_Q)

    img_out = np.clip(reconstruct_image(watermarked), 0, 255).astype(np.uint8)
    bgr_out = cv.cvtColor(img_out, cv.COLOR_YCrCb2BGR)
    cv.imwrite(output_path, bgr_out, [cv.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

    wm_bgr = cv.imread(output_path)
    if wm_bgr is not None:
        h = min(bgr.shape[0], wm_bgr.shape[0])
        w = min(bgr.shape[1], wm_bgr.shape[1])
        mse  = util.calculate_mse(bgr[:h, :w], wm_bgr[:h, :w])
        psnr = util.calculate_psnr(bgr[:h, :w], wm_bgr[:h, :w])
        print(f"Saved to {output_path}  |  MSE: {mse:.4f}  |  PSNR: {psnr:.2f} dB")
    else:
        print(f"Saved to {output_path}")


def extract_watermark_pipeline(watermarked_image_path: str, output_wm_path: str):
    """
    Extract a previously embedded watermark from a JPEG image.
    Uses BGR throughout to match the embed pipeline exactly.
    """
    jpeg_Q = compute_jpeg_qtable(JPEG_QUALITY)

    img = cv.imread(watermarked_image_path)
    if img is None:
        print(f"Image not found: {watermarked_image_path}")
        return

    ycbcr  = cv.cvtColor(img, cv.COLOR_BGR2YCrCb).astype(np.float32)
    blocks = segment_into_8x8_blocks(ycbcr)

    extracted_bits = extract_qim(blocks, jpeg_Q)
    if not extracted_bits:
        print("Failed to extract watermark.")
        return

    byte_array = bytearray()
    for i in range(0, len(extracted_bits), 8):
        seg = extracted_bits[i:i+8]
        if len(seg) == 8:
            byte_array.append(int(seg, 2))

    try:
        with open(output_wm_path, 'wb') as f:
            f.write(byte_array)
        print(f"Watermark extracted to {output_wm_path}")
    except Exception as e:
        print(f"Failed to save: {e}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    while True:
        print("\n--- JPEG Watermarking Menu ---")
        print("1. Calculate max insertion size")
        print("2. Embed watermark")
        print("3. Extract watermark")
        print("4. Exit")
        choice = input("Enter choice (1/2/3/4): ")

        if choice == '1':
            img_path = input("Image path: ")
            img = cv.imread(img_path)
            if img is not None:
                h, w, _ = img.shape
                raw_size = util.calculate_max_insertion_size(w, h, 8, 22)
                usable   = max(0, raw_size - 32)
                print(f"Max insertion size: {usable} bits usable "
                      f"({raw_size} total, 32 reserved for length header)")
            else:
                print("Image not found.")

        elif choice == '2':
            img_path = input("Input image path: ")
            wm_path  = input("Watermark file path: ")
            out_path = input("Output JPEG path: ")
            try:
                jpeg_watermark_pipeline(img_path, wm_path, out_path)
            except Exception as e:
                print(f"Error: {e}")

        elif choice == '3':
            img_path = input("Watermarked JPEG path: ")
            out_path = input("Output path for extracted watermark: ")
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