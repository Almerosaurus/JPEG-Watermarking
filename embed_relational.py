import cv2 as cv  
import numpy as np
import math

# ==========================================
# Functions originally from walk.py
# ==========================================
def findpoint(df, r):
    """
    Finds a sequence of coordinates in an 8x8 block using a zig-zag diagonal path
    to locate mid-frequency coefficients for embedding.
    """
    maxrow, maxcol = 8, 8
    # dire=-1 up-right, dire=1 down-left
    di = {-1: (-1, 1), 1: (1, -1)}
    xydict = {}
    while(r):	
        x, y, dire = df[0], df[1], df[2]
        # Handle boundaries
        if x == -1:
            df = (0, y - 2, 1)
            continue
        if y == -1:
            df = (x - 2, 0, -1)
            continue
        if x == maxrow:
            df = (maxrow - 1, y + 2, -1)
            continue
        if y == maxcol:
            df = (x + 2, maxcol - 1, 1)
            continue
            
        r = r - 1
        xydict[r] = (x, y)
        dx, dy = x + di[dire][0], y + di[dire][1]
        df = (dx, dy, dire)
    return xydict

def fpg(bsrc):
    """
    Generator that yields the (i, j) coordinates of the watermark image pixels one by one.
    """
    for i in range(bsrc.shape[0]):
        for j in range(bsrc.shape[1]):
            yield (i, j)
    # Yield (-1, -1) continuously to prevent StopIteration errors
    while True:
        yield (-1, -1)

# ==========================================
# Functions originally from embed.py
# ==========================================
def embed(watermark_path, host_path):
    # Read square watermark image (fingerprint) and invert colors
    src = cv.imread(watermark_path)
    if src is None:
        raise FileNotFoundError(f"Could not read watermark image at {watermark_path}")
    src = cv.bitwise_not(src)  
    
    # Convert to grayscale
    graysrc = cv.cvtColor(src, cv.COLOR_BGR2GRAY)  
    
    # Apply median filtering to reduce noise
    medianblurimg = cv.medianBlur(graysrc, 3)
    
    # Binarization: values < 70 become 255 (white), > 70 become 0 (black)
    ret, bsrc = cv.threshold(graysrc, 70, 255, 1)  
    cv.imshow('source', bsrc)
    cv.imwrite('embedfinger.jpg', bsrc, [int(cv.IMWRITE_JPEG_QUALITY), 100])

    # Read host image
    host = cv.imread(host_path)
    if host is None:
        raise FileNotFoundError(f"Could not read host image at {host_path}")
    
    # Convert host to YUV color space (embedding will be in the Y channel)
    hostyuv = cv.cvtColor(host, cv.COLOR_RGB2YUV)  
    # Convert to float32, which is required for DCT
    hostf = hostyuv.astype('float32')

    # Target output image
    finishwm = hostf
    
    # Create a matrix for observing 8x8 blocks
    wmblocks = np.zeros([hostf.shape[0], hostf.shape[1], 3], np.float32)
    wmblocks[:,:,:] = hostf[:,:,:]
    
    # Calculate the number of rows and columns for 8x8 blocks
    part8x8rownum = int(host.shape[0]/8)
    part8x8colnum = int(host.shape[1]/8)
    
    # Extract dimensions of the watermark to pass to the extractor later
    wm_height, wm_width = bsrc.shape[0], bsrc.shape[1]
    total_pixels = wm_height * wm_width
    
    # Calculate the average number of fingerprint pixels to store per 8x8 block
    r = math.ceil(total_pixels / (part8x8rownum * part8x8colnum))
    print(f"r = {r} (bits per 8x8 block)")
    
    # Find grid coordinates to use within each 8x8 block
    xydict = findpoint((3, 4, -1), r)
    
    # Fingerprint pixel generator
    fpgij = fpg(bsrc)
    
    count = 0
    flag = 0
    
    for parti in range(part8x8rownum):
        if flag: break
        for partj in range(part8x8colnum):
            if flag: break
            
            # Apply DCT to the 8x8 Y-channel block
            part8x8 = cv.dct(hostf[8*parti:8*parti+8, 8*partj:8*partj+8, 0])
            
            # Skip edge blocks that are smaller than 8x8
            if (part8x8.shape[0] < 8) or (part8x8.shape[1] < 8):
                continue
                
            # Embed 'r' pixels into this DCT block
            for t in range(r):
                if flag: break
                
                i, j = next(fpgij)
                if i == -1 and j == -1:
                    flag = 1
                
                rx, ry = xydict[t]
                
                # Relational embedding: adjust r1 and r2 (symmetric pair)
                r1 = part8x8[rx, ry]
                r2 = part8x8[7-rx, 7-ry] 
                
                detat = abs(r1 - r2)
                p = float(detat + 0.1) # Embedding depth/intensity
                
                if bsrc[i, j] == 0: # Black pixel
                    if r1 <= r2: 
                        part8x8[rx, ry] += p
                else: # White pixel
                    if r1 >= r2:
                        part8x8[7-rx, 7-ry] += p
                        
                if not flag:
                    count += 1
                    
            # Inverse DCT (IDCT)
            finishwm[8*parti:8*parti+8, 8*partj:8*partj+8, 0] = cv.idct(part8x8)
            wmblocks[8*parti:8*parti+8, 8*partj:8*partj+8, 0] = finishwm[8*parti:8*parti+8, 8*partj:8*partj+8, 0]
            
            # Draw block lines for the visualizer
            if (wmblocks.shape[0] > 8*parti+7) and (wmblocks.shape[1] > 8*partj+7):
                wmblocks[8*parti:8*parti+8, 8*partj+7, 0] = 100
                wmblocks[8*parti+7, 8*partj:8*partj+8, 0] = 100
                
    # Reconvert back to RGB color space
    wmrgb = cv.cvtColor(finishwm.astype('uint8'), cv.COLOR_YUV2RGB) 	
    
    cv.imshow('wmblocks', cv.cvtColor(wmblocks.astype('uint8'), cv.COLOR_YUV2RGB))
    cv.waitKey(0)  
    cv.destroyAllWindows()	
    
    # Save at different JPEG quality levels
    cv.imwrite("finishwm.jpg", wmrgb, [int(cv.IMWRITE_JPEG_QUALITY), 100])

    print("countembed =", count)
    
    # Return extraction parameters
    return r, wm_height, wm_width

# ==========================================
# Functions originally from extract.py
# ==========================================
def extract(src, dst, r, wm_height, wm_width):
    wmrgb = cv.imread(src)
    if wmrgb is None:
        raise FileNotFoundError(f"Could not read watermarked image at {src}")
        
    wmyuv = cv.cvtColor(wmrgb, cv.COLOR_RGB2YUV) 
    wmf = wmyuv.astype('float32')
    part8x8rownum = int(wmf.shape[0] / 8)
    part8x8colnum = int(wmf.shape[1] / 8)
    
    extractxydict = findpoint((3, 4, -1), r)
    
    # Maximum empty carrier for restored watermark matching original dimensions
    finishfinger = np.zeros([wm_height, wm_width, 3], np.uint8)
    i, j = 0, 0
    count = 0
    
    for parti in range(part8x8rownum):
        for partj in range(part8x8colnum):
            # Ignore blocks smaller than 8x8
            part8x8 = cv.dct(wmf[8*parti:8*parti+8, 8*partj:8*partj+8, 0])
            if (part8x8.shape[0] < 8) or (part8x8.shape[1] < 8):
                continue
            
            # Each 8x8 DCT block stores r fingerprint pixels
            for t in range(r):
                if i == wm_height:
                    break
                
                # The grid coordinates where the fingerprint pixel should be
                rx, ry = extractxydict[t]
                
                # Observe the relationship between r1 and r2 to determine if the pixel is black or white
                r1 = part8x8[rx, ry]
                r2 = part8x8[7-rx, 7-ry] # Centrally symmetric grid cell to r1
                
                if r1 > r2:
                    finishfinger[i, j] = 0 # Black
                elif r1 < r2:
                    finishfinger[i, j] = 255 # White
                    
                j += 1
                if j == wm_width:
                    j = 0
                    i += 1
                count += 1
                
    print(f"countextract ({dst}) =", count)
    cv.imwrite(dst, finishfinger, [int(cv.IMWRITE_JPEG_QUALITY), 100])

# ==========================================
# Main Execution
# ==========================================
def main():
    print("--- Image Watermarking Tool ---")
    host_path = input("Enter the path for the host image (e.g., 'host.jpg'): ").strip()
    wm_path = input("Enter the path for the watermark image (e.g., 'fingerprint.jpg'): ").strip()

    print("\n--- Starting Embedding Process ---")
    try:
        r, wm_height, wm_width = embed(wm_path, host_path)
    except Exception as e:
        print(f"Error during embedding: {e}")
        return

    print("\n--- Starting Extraction Process ---")
    extract("finishwm.jpg", "extractfinger.jpg", r, wm_height, wm_width)

    print("\nProcesses complete. Press any key on an image window to exit.")
    cv.waitKey(0)  
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()