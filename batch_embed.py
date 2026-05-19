import os
import sys
from PIL import Image
from QIM_embed import embed_watermark, extract_watermark

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def main():
    host_image = "input_self.jpg"
    watermark_image = "wm_rev.jpg"
    output_dir = "watermark_output"
    recovered_dir = "watermark_recovered"
    
    # Check if input files exist
    if not os.path.exists(host_image):
        print(f"Error: Host image '{host_image}' not found.")
        return
    if not os.path.exists(watermark_image):
        print(f"Error: Watermark image '{watermark_image}' not found.")
        return
        
    # Get watermark size
    with Image.open(watermark_image) as wm:
        wm_size = wm.size

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(recovered_dir, exist_ok=True)
    
    qualities = [90, 70, 50, 30, 10]
    
    print(f"Starting batch embedding and extraction...")
    
    for qf in qualities:
        output_filename = f"watermarked_qf{qf}.jpg"
        output_path = os.path.join(output_dir, output_filename)
        
        recovered_filename = f"recovered_qf{qf}.jpg"
        recovered_path = os.path.join(recovered_dir, recovered_filename)
        
        print(f"\n========================================")
        print(f" Processing Quality Factor: {qf}")
        print(f"========================================")
        
        try:
            # Embed
            print("[EMBEDDING]")
            embed_watermark(host_image, watermark_image, qf, output_path, delta_factor=2)
            
            # Extract
            print("\n[EXTRACTION]")
            extract_watermark(output_path, qf, wm_size, recovered_path, delta_factor=2, reference_path=watermark_image)
            
        except Exception as e:
            print(f"Failed to process QF {qf}: {e}")
            
    print(f"\nAll tasks completed. Check '{output_dir}' and '{recovered_dir}' folders.")

if __name__ == "__main__":
    main()
