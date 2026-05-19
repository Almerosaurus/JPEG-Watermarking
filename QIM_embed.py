import argparse
import os
import sys
import textwrap

import numpy as np
from PIL import Image
from scipy.fft import dctn, idctn

LUMA_BASE = np.array([
    [16, 11, 10, 16,  24,  40,  51,  61],
    [12, 12, 14, 19,  26,  58,  60,  55],
    [14, 13, 16, 24,  40,  57,  69,  56],
    [14, 17, 22, 29,  51,  87,  80,  62],
    [18, 22, 37, 56,  68, 109, 103,  77],
    [24, 35, 55, 64,  81, 104, 113,  92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103,  99],
], dtype=np.float64)

CHROMA_BASE = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
], dtype=np.float64)

EMBED_POSITIONS = [(2, 2)]
BITS_PER_BLOCK  = len(EMBED_POSITIONS)

def scale_qtable(base: np.ndarray, quality: int) -> np.ndarray:
    
    quality = int(np.clip(quality, 1, 100))
    scale   = 5000.0 / quality if quality < 50 else 200.0 - 2.0 * quality
    table   = np.floor((base * scale + 50.0) / 100.0)
    return np.clip(table, 1, 255).astype(np.int32)

def pad_to_multiple(arr: np.ndarray, multiple: int = 8):
    
    h, w    = arr.shape
    pad_h   = (-h) % multiple
    pad_w   = (-w) % multiple
    padded  = np.pad(arr, ((0, pad_h), (0, pad_w)), mode="edge")
    return padded, pad_h, pad_w

def dct2(block: np.ndarray) -> np.ndarray:
    
    return dctn(block, norm="ortho")

def idct2(block: np.ndarray) -> np.ndarray:
    
    return idctn(block, norm="ortho")

def rgb_to_ycbcr(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    Y  =  0.299000 * R + 0.587000 * G + 0.114000 * B
    Cb = -0.168736 * R - 0.331264 * G + 0.500000 * B + 128.0
    Cr =  0.500000 * R - 0.418688 * G - 0.081312 * B + 128.0
    return Y, Cb, Cr

def ycbcr_to_rgb(Y: np.ndarray,
                 Cb: np.ndarray,
                 Cr: np.ndarray) -> np.ndarray:
    
    Cb_ = Cb - 128.0
    Cr_ = Cr - 128.0
    R   = Y                      + 1.402000 * Cr_
    G   = Y - 0.344136 * Cb_    - 0.714136 * Cr_
    B   = Y + 1.772000 * Cb_
    return np.clip(np.stack([R, G, B], axis=-1), 0.0, 255.0)

def watermark_to_bits(wm_img: Image.Image) -> np.ndarray:
    
    arr = np.array(wm_img, dtype=np.uint8)
    return (arr >= 128).astype(np.uint8).ravel()

def qim_embed_coeff(F: float, delta: float, bit: int) -> float:
    
    if bit == 0:
        return delta * round(F / delta)
    else:
        return delta * round((F - delta * 0.5) / delta) + delta * 0.5

def qim_extract_coeff(q: int, delta_factor: int) -> int:
    
    half = delta_factor / 2.0
    return int(round(q / half)) % 2

def process_luma_embed(channel:      np.ndarray,
                       qtable:       np.ndarray,
                       wm_bits:      np.ndarray,
                       delta_factor: int) -> tuple[np.ndarray, int]:
    
    padded, _, _ = pad_to_multiple(channel)
    H, W         = padded.shape
    output       = np.zeros_like(padded)

    bit_idx  = 0
    total    = len(wm_bits)
    qtable_f = qtable.astype(np.float64)

    for row in range(0, H, 8):
        for col in range(0, W, 8):
            block     = padded[row:row+8, col:col+8] - 128.0
            dct_block = dct2(block)

            dct_qim = dct_block.copy()
            for (r, c) in EMBED_POSITIONS:
                if bit_idx >= total:
                    break
                delta          = float(delta_factor) * qtable_f[r, c]
                dct_qim[r, c] = qim_embed_coeff(dct_block[r, c], delta,
                                                 int(wm_bits[bit_idx]))
                bit_idx += 1

            q_block = np.round(dct_qim / qtable_f).astype(np.int32)

            deq_block = q_block.astype(np.float64) * qtable_f
            output[row:row+8, col:col+8] = idct2(deq_block) + 128.0

    return output, bit_idx

def process_channel_compress(channel: np.ndarray,
                             qtable:  np.ndarray) -> np.ndarray:
    
    padded, _, _ = pad_to_multiple(channel)
    H, W         = padded.shape
    output       = np.zeros_like(padded)
    qtable_f     = qtable.astype(np.float64)

    for row in range(0, H, 8):
        for col in range(0, W, 8):
            block    = padded[row:row+8, col:col+8] - 128.0
            dct_blk  = dct2(block)
            q_blk    = np.round(dct_blk / qtable_f).astype(np.int32)
            deq_blk  = q_blk.astype(np.float64) * qtable_f
            output[row:row+8, col:col+8] = idct2(deq_blk) + 128.0

    return output

def embed_watermark(image_path:     str,
                    watermark_path: str,
                    quality:        int,
                    output_path:    str,
                    delta_factor:   int = 2) -> None:
    
    img      = Image.open(image_path).convert("RGB")
    img_arr  = np.array(img, dtype=np.float64)
    H_orig, W_orig = img_arr.shape[:2]

    wm       = Image.open(watermark_path).convert("L")
    wm_bits  = watermark_to_bits(wm)
    wm_h, wm_w = np.array(wm).shape

    total_bits   = len(wm_bits)
    blocks_avail = ((H_orig + 7) // 8) * ((W_orig + 7) // 8)
    capacity     = blocks_avail * BITS_PER_BLOCK

    luma_q   = scale_qtable(LUMA_BASE,   quality)
    chroma_q = scale_qtable(CHROMA_BASE, quality)

    embed_deltas = {(r, c): delta_factor * int(luma_q[r, c])
                    for (r, c) in EMBED_POSITIONS}

    sep = "─" * 60
    print(sep)
    print("  QIM Watermark Embedding — Mid-Frequency DCT Coefficients")
    print(sep)
    print(f"  Image        : {image_path}  ({W_orig}×{H_orig} px)")
    print(f"  Watermark    : {watermark_path}  ({wm_w}×{wm_h} px, {total_bits} bits)")
    print(f"  Quality      : {quality}")
    print(f"  delta_factor : {delta_factor}  "
          f"(QIM step Δ[r,c] = {delta_factor} × Q[r,c])")
    print(f"  Embed pos    : {EMBED_POSITIONS}  ({BITS_PER_BLOCK} bits/block)")
    print(f"  QIM steps    : ", end="")
    print("  ".join(f"({r},{c})→Δ={v}" for (r, c), v in embed_deltas.items()))
    print(f"  Extraction   : b̂ = round(q / {delta_factor // 2}) mod 2")
    print(f"  Blocks       : {blocks_avail}  →  capacity = {capacity} bits")
    print(f"  Output       : {output_path}")
    print()
    print("  Scaled luminance Q-table:")
    for row in luma_q:
        print("    " + "  ".join(f"{v:3d}" for v in row))
    print()

    if total_bits > capacity:
        print(f"  [WARNING] Watermark ({total_bits} bits) exceeds capacity "
              f"({capacity} bits).\n"
              f"            Only the first {capacity} bits will be embedded.\n")

    Y, Cb, Cr = rgb_to_ycbcr(img_arr)

    print("  [1/4] QIM embedding into Y channel …")
    Y_wm, embedded = process_luma_embed(Y, luma_q, wm_bits, delta_factor)

    print("  [2/4] Compressing Cb channel …")
    Cb_c = process_channel_compress(Cb, chroma_q)

    print("  [3/4] Compressing Cr channel …")
    Cr_c = process_channel_compress(Cr, chroma_q)

    Y_wm  = Y_wm [:H_orig, :W_orig]
    Cb_c  = Cb_c [:H_orig, :W_orig]
    Cr_c  = Cr_c [:H_orig, :W_orig]

    rgb_out = ycbcr_to_rgb(Y_wm, Cb_c, Cr_c)
    out_img = Image.fromarray(rgb_out.astype(np.uint8), "RGB")

    pillow_qtables = {
        0: luma_q.flatten().tolist(),
        1: chroma_q.flatten().tolist(),
    }
    print("  [4/4] Saving output JPEG with custom Q-table …")
    out_img.save(output_path, "JPEG",
                 qtables=pillow_qtables,
                 subsampling=0)
    in_size  = os.path.getsize(image_path)
    out_size = os.path.getsize(output_path)
    
    try:
        from utils import calculate_mse_psnr
        mse, psnr = calculate_mse_psnr(image_path, output_path)
    except ImportError:
        mse, psnr = None, None

    print()
    print(sep)
    print(f"  Bits embedded    : {embedded} / {total_bits}")
    print(f"  delta_factor     : {delta_factor}  "
          f"(max distortion per coeff = Δ/4 = Q[r,c] × {delta_factor}/4)")
    print(f"  Input file size  : {in_size:,} bytes")
    print(f"  Output file size : {out_size:,} bytes  "
          f"({out_size/in_size*100:.1f}% of input)")
    if mse is not None:
        print(f"  MSE              : {mse:.4f}")
        print(f"  PSNR             : {psnr:.2f} dB")
    print(f"  Done → {output_path}")
    print(sep)
    return mse, psnr

def extract_luma_bits(channel:      np.ndarray,
                      qtable:       np.ndarray,
                      n_bits:       int,
                      delta_factor: int) -> np.ndarray:
    
    padded, _, _ = pad_to_multiple(channel)
    H, W         = padded.shape
    qtable_f     = qtable.astype(np.float64)
    half         = delta_factor / 2.0

    bits    = []
    bit_idx = 0

    for row in range(0, H, 8):
        for col in range(0, W, 8):
            if bit_idx >= n_bits:
                break
            block     = padded[row:row+8, col:col+8] - 128.0
            dct_block = dct2(block)

            for (r, c) in EMBED_POSITIONS:
                if bit_idx >= n_bits:
                    break
                q_int = int(round(dct_block[r, c] / qtable_f[r, c]))
                bits.append(int(round(q_int / half)) % 2)
                bit_idx += 1
        if bit_idx >= n_bits:
            break

    return np.array(bits, dtype=np.uint8)

def majority_vote(bits: np.ndarray,
                  wm_h: int, wm_w: int) -> np.ndarray:
    
    binary = bits[:wm_h * wm_w].reshape(wm_h, wm_w)
    return (binary * 255).astype(np.uint8)

def extract_watermark(watermarked_path: str,
                      quality:          int,
                      wm_size:          tuple[int, int],
                      output_path:      str,
                      delta_factor:     int = 2,
                      reference_path:   str | None = None) -> None:
    
    wm_w, wm_h   = wm_size
    n_bits        = wm_w * wm_h
    blocks_needed = (n_bits + BITS_PER_BLOCK - 1) // BITS_PER_BLOCK

    img     = Image.open(watermarked_path).convert("RGB")
    img_arr = np.array(img, dtype=np.float64)
    H_img, W_img = img_arr.shape[:2]

    blocks_avail = ((H_img + 7) // 8) * ((W_img + 7) // 8)

    luma_q = scale_qtable(LUMA_BASE, quality)

    embed_deltas = {(r, c): delta_factor * int(luma_q[r, c])
                    for (r, c) in EMBED_POSITIONS}

    sep = "─" * 60
    print(sep)
    print("  QIM Watermark Extraction — Mid-Frequency DCT Coefficients")
    print(sep)
    print(f"  Watermarked  : {watermarked_path}  ({W_img}×{H_img} px)")
    print(f"  Watermark sz : {wm_w}×{wm_h} px  ({n_bits} bits to extract)")
    print(f"  Quality      : {quality}")
    print(f"  delta_factor : {delta_factor}  "
          f"(Δ[r,c] = {delta_factor} × Q[r,c])")
    print(f"  Embed pos    : {EMBED_POSITIONS}")
    print(f"  QIM steps    : ", end="")
    print("  ".join(f"({r},{c})→Δ={v}" for (r, c), v in embed_deltas.items()))
    print(f"  Decision rule: b̂ = round(q / {delta_factor // 2}) mod 2")
    print(f"  Blocks avail : {blocks_avail}  (need ≥ {blocks_needed})")
    print(f"  Output       : {output_path}")
    if reference_path:
        print(f"  Reference WM : {reference_path}  (for BER computation)")
    print()

    if blocks_avail < blocks_needed:
        print(f"  [ERROR] Image is too small to hold a {wm_w}×{wm_h} watermark "
              f"at these settings.\n"
              f"          Need {blocks_needed} blocks, have {blocks_avail}.",
              file=sys.stderr)
        sys.exit(1)

    Y, _, _ = rgb_to_ycbcr(img_arr)

    print("  [1/3] Extracting bits from Y channel …")
    bits = extract_luma_bits(Y, luma_q, n_bits, delta_factor)

    print("  [2/3] Reconstructing watermark image …")
    wm_img_arr = majority_vote(bits, wm_h, wm_w)
    wm_out     = Image.fromarray(wm_img_arr, mode="L")

    print("  [3/3] Saving recovered watermark …")
    wm_out.save(output_path)

    print()
    print(sep)
    print(f"  Bits extracted : {len(bits)}")

    if reference_path:
        try:
            from utils import calculate_ber
            errors, ber, total = calculate_ber(reference_path, output_path)
            correct = total - errors

            print(f"  Bits correct   : {correct} / {total}")
            print(f"  Bit errors     : {errors}")
            print(f"  BER            : {ber:.4f}  ({ber*100:.2f}%)")
            print(f"  Accuracy       : {(1-ber)*100:.2f}%")
        except ImportError:
            ref     = Image.open(reference_path).convert("L")
            ref_arr = np.array(ref.resize((wm_w, wm_h), Image.NEAREST), dtype=np.uint8)
            ref_bits = (ref_arr >= 128).astype(np.uint8).ravel()[:n_bits]

            errors  = int(np.sum(bits != ref_bits))
            ber     = errors / n_bits
            correct = n_bits - errors

            print(f"  Bits correct   : {correct} / {n_bits}")
            print(f"  Bit errors     : {errors}")
            print(f"  BER            : {ber:.4f}  ({ber*100:.2f}%)")
            print(f"  Accuracy       : {(1-ber)*100:.2f}%")
        result = (errors, ber, correct, n_bits)
    else:
        result = None

    print(f"  Done → {output_path}")
    print(sep)
    return result

def parse_args():
    root = argparse.ArgumentParser(
        prog="qim_watermark.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="",
    )
    sub = root.add_subparsers(dest="cmd", metavar="subcommand")
    sub.required = True

    def add_common(p):
        p.add_argument("quality", type=int,
                       help="JPEG quality factor used during embedding (1–100)")
        p.add_argument("--delta", type=int, default=2, metavar="EVEN_INT",
                       help="")

    pe = sub.add_parser(
        "embed",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Compress image and embed watermark via QIM",
        description="",
        epilog="",
    )
    pe.add_argument("image",
                    help="Input image (JPEG/PNG/BMP …)")
    pe.add_argument("watermark",
                    help="Watermark image — loaded as grayscale, "
                         "binarised at threshold 128")
    add_common(pe)
    pe.add_argument("-o", "--output", default="watermarked.jpg",
                    metavar="OUTPUT",
                    help="Output JPEG path  (default: watermarked.jpg)")

    px = sub.add_parser(
        "extract",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Recover the watermark from a watermarked image",
        description="",
        epilog="",
    )
    px.add_argument("watermarked",
                    help="Watermarked JPEG to read from")
    add_common(px)
    px.add_argument("--wm-size", required=True, metavar="WxH",
                    help="Watermark dimensions used at embed time, e.g. 64x64")
    px.add_argument("-o", "--output", default="extracted_wm.jpg",
                    metavar="OUTPUT",
                    help="Output path for recovered watermark  "
                         "(default: extracted_wm.jpg)")
    px.add_argument("--ref", default=None, metavar="REFERENCE",
                    help="Original watermark image for BER computation "
                         "(optional)")

    pa = sub.add_parser("attack", help="Perform a recompression attack on a watermarked JPEG")
    pa.add_argument("watermarked", help="Watermarked JPEG to read from")
    pa.add_argument("output", help="Output path for attacked JPEG")
    pa.add_argument("quality", type=int, help="JPEG quality factor for recompression (1-100)")

    pc = sub.add_parser("capacity", help="Calculate max watermark insertion size for an image")
    pc.add_argument("image", help="Input image")

    return root.parse_args()

def main():
    args = parse_args()

    if hasattr(args, 'quality') and not (1 <= args.quality <= 100):
        print("Error: quality must be between 1 and 100.", file=sys.stderr)
        sys.exit(1)
    if hasattr(args, 'delta') and (args.delta < 2 or args.delta % 2 != 0):
        print("Error: --delta must be an even integer ≥ 2 (e.g. 2, 4, 6).",
              file=sys.stderr)
        sys.exit(1)

    if args.cmd == "embed":
        for path, label in [(args.image, "image"),
                            (args.watermark, "watermark")]:
            if not os.path.exists(path):
                print(f"Error: {label} file not found: {path}", file=sys.stderr)
                sys.exit(1)
        embed_watermark(args.image, args.watermark, args.quality,
                        args.output, args.delta)

    elif args.cmd == "extract":
        if not os.path.exists(args.watermarked):
            print(f"Error: watermarked file not found: {args.watermarked}",
                  file=sys.stderr)
            sys.exit(1)
        if args.ref and not os.path.exists(args.ref):
            print(f"Error: reference file not found: {args.ref}", file=sys.stderr)
            sys.exit(1)

        try:
            wm_w_str, wm_h_str = args.wm_size.lower().split("x")
            wm_size = (int(wm_w_str), int(wm_h_str))
            if wm_size[0] < 1 or wm_size[1] < 1:
                raise ValueError
        except ValueError:
            print("Error: --wm-size must be in WxH format, e.g. 64x64",
                  file=sys.stderr)
            sys.exit(1)

        extract_watermark(args.watermarked, args.quality, wm_size,
                          args.output, args.delta, args.ref)
                          
    elif args.cmd == "attack":
        if not os.path.exists(args.watermarked):
            print(f"Error: file not found: {args.watermarked}", file=sys.stderr)
            sys.exit(1)
        from utils import recompress_attack
        recompress_attack(args.watermarked, args.output, args.quality)

    elif args.cmd == "capacity":
        if not os.path.exists(args.image):
            print(f"Error: file not found: {args.image}", file=sys.stderr)
            sys.exit(1)
        from utils import calculate_max_insertion_size
        max_bits = calculate_max_insertion_size(args.image)
        print(f"Max insertion capacity for {args.image}: {max_bits} bits")

if __name__ == "__main__":
    main()