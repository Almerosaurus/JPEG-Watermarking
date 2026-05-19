# JPEG-Watermarking

A robust Python-based JPEG image watermarking pipeline using Quantization Index Modulation (QIM) in the Mid-Frequency DCT domain. This project allows you to embed binary watermarks (like small images) into JPEG files and extract them later with high accuracy. It also includes tools for testing the robustness of the watermark against JPEG compression attacks.

## Features
- **QIM Embedding**: Embeds watermark bits into the mid-frequency DCT coefficients of the Luminance (Y) channel.
- **Robustness Testing**: Includes batch processing scripts to test watermark survivability across different JPEG quality factors (e.g., QF 90 down to QF 10).
- **Evaluation Metrics**: Automatically calculates Bit Error Rate (BER), Mean Squared Error (MSE), and Peak Signal-to-Noise Ratio (PSNR).
- **Interactive & CLI Modes**: Use the interactive menu or the command-line interface for your workflows.

## Installation

Install the required dependencies using `pip`:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Command-Line Interface (`QIM_embed.py`)

The main CLI tool provides several subcommands for different operations.

**Check Capacity**
Check how many bits a host image can hold:
```bash
python QIM_embed.py capacity input_self.jpg
```

**Embed Watermark**
Embed a binary watermark image into a host image:
```bash
python QIM_embed.py embed input_self.jpg wm_rev.jpg 90 -o watermarked.jpg
```

**Extract Watermark**
Extract the watermark from a JPEG image (requires knowing the original watermark size). You can also pass the original watermark as a reference to compute BER:
```bash
python QIM_embed.py extract watermarked.jpg --wm-size 150x150 -o extracted_wm.jpg --ref wm_rev.jpg
```

**Recompression Attack**
Test robustness by simulating a JPEG recompression attack:
```bash
python QIM_embed.py attack watermarked.jpg attacked_output.jpg 50
```

### 2. Batch Robustness Testing (`batch_embed.py`)

Run the automated batch script to embed and extract watermarks across multiple quality factors (90, 70, 50, 30, 10). This script is perfect for generating data for reports.

```bash
python batch_embed.py
```
- Output images are saved in the `watermark_output` folder.
- Extracted watermarks are saved in the `watermark_recovered` folder.

### 3. Interactive Menu (`watermark.py`)

If you prefer a guided experience, run the interactive menu:
```bash
python watermark.py
```
This will launch a prompt allowing you to calculate capacity, embed, and extract watermarks step-by-step.

## Project Structure

- `QIM_embed.py`: Advanced CLI tool for QIM embedding, extraction, and attacks.
- `batch_embed.py`: Automation script for testing robustness across different quality factors.
- `utils.py`: Utility functions for BER, MSE, PSNR, and capacity calculations.
- `jpeg_encoder.py`: An experimental, pure-Python manual JPEG encoder that writes raw quantized DCT blocks directly to the bitstream.