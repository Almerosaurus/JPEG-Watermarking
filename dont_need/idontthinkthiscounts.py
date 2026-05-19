import numpy as np
from PIL import Image
import scipy.fftpack as fftpack
import struct
import io

# Standard JPEG Luminance Quantization Table (Quality 50)
Q_LUM = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=np.float32)

# Standard JPEG Chrominance Quantization Table (Quality 50)
Q_CHR = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
], dtype=np.float32)

ZIGZAG_ORDER = [
    0, 1, 8, 16, 9, 2, 3, 10,
    17, 24, 32, 25, 18, 11, 4, 5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63
]

def get_quantization_tables(quality):
    """Adjust standard quantization tables based on desired quality factor."""
    if quality <= 0: quality = 1
    if quality > 100: quality = 100
    
    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - quality * 2
        
    q_lum_scaled = np.floor((Q_LUM * scale + 50) / 100)
    q_chr_scaled = np.floor((Q_CHR * scale + 50) / 100)
    
    q_lum_scaled = np.clip(q_lum_scaled, 1, 255).astype(np.uint8)
    q_chr_scaled = np.clip(q_chr_scaled, 1, 255).astype(np.uint8)
    
    return q_lum_scaled, q_chr_scaled

def rgb_to_ycbcr(image):
    """Convert RGB image to YCbCr color space."""
    r = image[:,:,0]
    g = image[:,:,1]
    b = image[:,:,2]
    
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.1687 * r - 0.3313 * g + 0.5 * b + 128
    cr = 0.5 * r - 0.4187 * g - 0.0813 * b + 128
    
    return y, cb, cr

def block_dct(block):
    """Apply 2D Discrete Cosine Transform to an 8x8 block."""
    # JPEG centers values around 0 by subtracting 128
    centered_block = block - 128.0
    return fftpack.dct(fftpack.dct(centered_block.T, norm='ortho').T, norm='ortho')

def quantize_and_zigzag(dct_block, quant_table):
    """Quantize the DCT block and order values in zigzag sequence."""
    quantized = np.round(dct_block / quant_table).astype(np.int32)
    flat = quantized.flatten()
    zigzag_block = [flat[i] for i in ZIGZAG_ORDER]
    return zigzag_block

def write_jpeg_header(f, width, height, watermark_data, q_lum, q_chr):
    """Write necessary JPEG markers and headers, including the watermark image."""
    # Start of Image (SOI)
    f.write(bytes([0xFF, 0xD8]))
    
    # APP0 Segment (JFIF)
    f.write(bytes([0xFF, 0xE0]))
    f.write(struct.pack(">H", 16)) # Length
    f.write(b"JFIF\x00") # Identifier
    f.write(bytes([0x01, 0x01])) # Version
    f.write(bytes([0x00])) # Units (0 = no units)
    f.write(struct.pack(">H", 1)) # X density
    f.write(struct.pack(">H", 1)) # Y density
    f.write(bytes([0x00, 0x00])) # Thumbnail
    
    # Custom APP1 Segment(s) for WATERMARK IMAGE
    # Since JPEG markers max out at 65535 bytes, we chunk the watermark image
    # into multiple APP1 markers if it's larger than ~65KB.
    if watermark_data:
        max_chunk_size = 65533 # 65535 - 2 bytes for length header
        for i in range(0, len(watermark_data), max_chunk_size):
            chunk = watermark_data[i:i+max_chunk_size]
            length = 2 + len(chunk)
            f.write(bytes([0xFF, 0xE1])) # APP1 marker
            f.write(struct.pack(">H", length))
            f.write(chunk)

    # Define Quantization Table (DQT) - Luminance
    f.write(bytes([0xFF, 0xDB]))
    f.write(struct.pack(">H", 67)) # Length (2 bytes + 1 byte info + 64 bytes table)
    f.write(bytes([0x00])) # Table info (0 = Lum, 8-bit)
    f.write(q_lum.flatten().tobytes())

    # Define Quantization Table (DQT) - Chrominance
    f.write(bytes([0xFF, 0xDB]))
    f.write(struct.pack(">H", 67)) # Length
    f.write(bytes([0x01])) # Table info (1 = Chr, 8-bit)
    f.write(q_chr.flatten().tobytes())

    # Start of Frame (SOF0) - Baseline DCT
    f.write(bytes([0xFF, 0xC0]))
    f.write(struct.pack(">H", 17)) # Length
    f.write(bytes([0x08])) # Precision (8-bit)
    f.write(struct.pack(">H", height))
    f.write(struct.pack(">H", width))
    f.write(bytes([0x03])) # Number of components (3 for YCbCr)
    
    # Component info: ID, sampling factors, quant table ID
    f.write(bytes([0x01, 0x11, 0x00])) # Y
    f.write(bytes([0x02, 0x11, 0x01])) # Cb
    f.write(bytes([0x03, 0x11, 0x01])) # Cr

    # Define Huffman Table (DHT) - Dummy tables just to make it a valid format
    # In a real encoder, you would calculate these based on the actual data frequencies
    # or use standard tables. For this structural example, we write minimal valid markers.
    # Note: A real encoder MUST write actual huffman tables.
    _write_standard_huffman_tables(f)

    # Start of Scan (SOS)
    f.write(bytes([0xFF, 0xDA]))
    f.write(struct.pack(">H", 12)) # Length
    f.write(bytes([0x03])) # Number of components
    f.write(bytes([0x01, 0x00])) # Y uses table 0
    f.write(bytes([0x02, 0x11])) # Cb uses table 1
    f.write(bytes([0x03, 0x11])) # Cr uses table 1
    f.write(bytes([0x00, 0x3F, 0x00])) # Spectral selection, successive approx

def _write_standard_huffman_tables(f):
    """Write standard JPEG Huffman tables."""
    # Standard DC Luminance
    f.write(bytes([0xFF, 0xC4]))
    dc_lum_counts = bytes([0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    dc_lum_symbols = bytes([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    f.write(struct.pack(">H", 2 + 1 + 16 + len(dc_lum_symbols)))
    f.write(bytes([0x00])) # Class 0 (DC), Table 0
    f.write(dc_lum_counts)
    f.write(dc_lum_symbols)

    # (Simplified: typically you need AC Lum, DC Chr, AC Chr tables as well)
    # For a fully functional encoder from scratch, implementing the bitstream
    # generation for Huffman coding is extremely complex and usually where
    # libraries are strictly required. This script demonstrates the *structure*
    # up to the point of encoding the bitstream, focusing on the header/watermark request.

def compress_and_watermark(image_path, output_path, watermark_image_path=None, quality=50):
    # Let Pillow encode the real JPEG
    img = Image.open(image_path).convert('RGB')
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=quality)
    jpeg_bytes = buffer.getvalue()

    # Load watermark
    watermark_data = b""
    if watermark_image_path:
        with open(watermark_image_path, 'rb') as wf:
            watermark_data = wf.read()

    # Inject APP1 chunks right after the SOI marker (first 2 bytes)
    soi = jpeg_bytes[:2]          # FF D8
    rest = jpeg_bytes[2:]         # everything else

    app1_chunks = b""
    max_chunk = 65533
    for i in range(0, len(watermark_data), max_chunk):
        chunk = watermark_data[i:i+max_chunk]
        length = 2 + len(chunk)
        app1_chunks += bytes([0xFF, 0xE1])
        app1_chunks += struct.pack(">H", length)
        app1_chunks += chunk

    with open(output_path, 'wb') as f:
        f.write(soi + app1_chunks + rest)

    print(f"Done. Watermark embedded: {len(watermark_data)} bytes.")
    
def extract_watermark(watermarked_image_path, output_watermark_path):
    """
    Parses a JPEG file to find custom APP1 markers and extracts 
    the embedded binary payload (the watermark image).
    """
    print(f"Extracting watermark from {watermarked_image_path}...")
    try:
        with open(watermarked_image_path, 'rb') as f:
            data = f.read()
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # A valid JPEG must start with Start of Image (SOI) marker: FF D8
    if data[0:2] != b'\xff\xd8':
        print("Error: The provided file is not a valid JPEG.")
        return

    watermark_data = bytearray()
    idx = 2 # Start after SOI

    # Parse JPEG markers
    while idx < len(data):
        # Look for the marker identifier (0xFF)
        if data[idx] != 0xFF:
            # If we lose the marker sync in the header, something is wrong
            break

        marker = data[idx+1]
        idx += 2

        # Standalone markers that have no length data
        if marker in [0xD8, 0xD9, 0x00] or (0xD0 <= marker <= 0xD7):
            if marker == 0xD9: # End of Image (EOI)
                break
            continue

        # All other standard headers have a 2-byte length immediately following the marker
        length = struct.unpack(">H", data[idx:idx+2])[0]

        # 0xE1 is our APP1 marker where we stored the watermark chunks
        if marker == 0xE1:
            # The length includes the 2 bytes of the length field itself
            payload = data[idx+2 : idx+length]
            watermark_data.extend(payload)
            print(f"Found APP1 chunk: {len(payload)} bytes")

        # 0xDA is Start of Scan (SOS). After this, the compressed image bitstream begins.
        # Since our watermark is entirely in the headers, we can stop parsing here.
        if marker == 0xDA:
            break

        # Move to the next marker
        idx += length

    # Validate if what we extracted looks like a JPEG (starts with FF D8)
    if len(watermark_data) > 0 and watermark_data.startswith(b'\xff\xd8'):
        try:
            with open(output_watermark_path, 'wb') as wf:
                wf.write(watermark_data)
            print(f"Success! Watermark saved to {output_watermark_path} ({len(watermark_data)} bytes).")
        except Exception as e:
            print(f"Error saving extracted watermark: {e}")
    else:
        print("No valid JPEG watermark found in the headers.")

if __name__ == "__main__":
    # To test this, you would need a sample image named 'input.jpg' in the same directory.
    # We will wrap it in a try-catch for demonstration purposes.
    
    print("--- JPEG Compressor with Header Watermark ---")
    print("This script demonstrates the structure of an incomplete JPEG encoder,")
    print("focusing on DCT, quantization, and embedding custom markers in the header.")
    
    # compress_and_watermark('input.jpg', 'output_watermarked.jpg', watermark_image_path='wm.jpg', quality=50)
    
    extract_watermark('output_watermarked-q75.jpg', 'recovered_watermark.jpg')

    