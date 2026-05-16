import cv2 as cv  
import numpy as np
import math

# ==========================================
# Functions originally from walk.py
# ==========================================
def findpoint(df, r):
    """
    Finds a sequence of coordinates in an 8x8 block using a zig-zag diagonal path.
    This helps target mid-frequency coefficients for embedding.
    """
    maxrow, maxcol = 8, 8
    # dire=-1 means up-right, dire=1 means down-left
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
    # Yield (-1, -1) continuously to prevent StopIteration errors when done
    while True:
        yield (-1, -1)

# ==========================================
# Functions originally from embed.py
# ==========================================
def embed(srcs):
    # Read square watermark image (fingerprint) and invert it
    src = cv.imread(srcs)    
    src = cv.bitwise_not(src)  
    
    # Convert to grayscale
    graysrc = cv.cvtColor(src, cv.COLOR_BGR2GRAY)  
    
    # Apply median filtering
    medianblurimg = cv.medianBlur(graysrc, 3)
    
    # Binarization: values < 70 become 255 (white), > 70 become 0 (black)
    ret, bsrc = cv.threshold(graysrc, 70, 255, 1)  
    cv.imshow('source', bsrc)
    cv.imwrite('embedfinger.jpg', bsrc, [int(cv.IMWRITE_JPEG_QUALITY), 100])

    # Read host image
    host = cv.imread('host.jpg')  
    # Convert host to YUV (we will embed in the Y channel)
    hostyuv = cv.cvtColor(host, cv.COLOR_RGB2YUV)  
    # Convert to float32, which is required for DCT
    hostf = hostyuv.astype('float32')

    # Target finish image
    finishwm = hostf
    
    # Create visualization for 8x8 blocks
    wmblocks = np.zeros([hostf.shape[0], hostf.shape[1], 3], np.float32)
    wmblocks[:,:,:] = hostf[:,:,:]
    
    # Calculate row/col counts for 8x8 blocks
    part8x8rownum = int(host.shape[0]/8)
    part8x8colnum = int(host.shape[1]/8)
    
    # Total pixels in the fingerprint
    fingernum = bsrc.shape[0] * bsrc.shape[1]
    
    # Calculate average number of fingerprint pixels to store per 8x8 block
    r = math.ceil(fingernum / (part8x8rownum * part8x8colnum))
    print("r=", r)
    
    # Find grid coordinates to use within each 8x8 block
    xydict = findpoint((3, 4, -1), r)
    
    # Generator for fingerprint pixels
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
                
                # Relational embedding: adjust r1 and r2 (symmetric pairs)
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
                    
            # Inverse DCT
            finishwm[8*parti:8*parti+8, 8*partj:8*partj+8, 0] = cv.idct(part8x8)
            wmblocks[8*parti:8*parti+8, 8*partj:8*partj+8, 0] = finishwm[8*parti:8*parti+8, 8*partj:8*partj+8, 0]
            
            # Draw block lines for visualizer
            if (wmblocks.shape[0] > 8*parti+7) and (wmblocks.shape[1] > 8*partj+7):
                wmblocks[8*parti:8*parti+8, 8*partj+7, 0] = 100
                wmblocks[8*parti+7, 8*partj:8*partj+8, 0] = 100
                
    # Reconvert to RGB space
    wmrgb = cv.cvtColor(finishwm.astype('uint8'), cv.COLOR_YUV2RGB) 	
    
    cv.imshow('wmblocks', cv.cvtColor(wmblocks.astype('uint8'), cv.COLOR_YUV2RGB))
    cv.waitKey(0)  
    cv.destroyAllWindows()	
    
    # Save at different JPEG quality levels
    for x in range(6):
        name = "finishwm" + str(x)
        filename = name + ".jpg"
        cv.imwrite(filename, wmrgb, [int(cv.IMWRITE_JPEG_QUALITY), 100-x])
        
        img = cv.imread(filename)
        cv.namedWindow(name, 0)	
        k = 480
        cv.resizeWindow(name, k, int(k * img.shape[0] / img.shape[1]))
        cv.imshow(name, img)

    print("countembed=", count)

def main():
    try:
        embed('fingerprint.jpg')
    except Exception as e:
        print(f"Error reading image: {e}")
    cv.waitKey(0)  
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()