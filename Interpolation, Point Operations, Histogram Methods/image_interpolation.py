import cv2
import numpy as np

# LOAD GRAYSCALE LENA IMAGE
lena = cv2.imread("C:\\Users\\imanc\\Documents\\FALL24\\CP467-ImageProccesing\\A1\\A1_chau0820\\Images\\lena.tif", 0)
h, w = lena.shape[:2]

# NEAREST NEIGHBOUR scratch
def nn_interpolation (img, scale):
    h, w = img.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)
    resized_img = np.zeros((new_h, new_w), dtype=np.uint8)
    
    for x in range(new_h):
        for y in range(new_w):
            a = int(x/scale)
            b = int(y/scale)
            resized_img[x,y] = img[a,b]
    return resized_img

downscaled_nn_scratch = nn_interpolation(lena, 0.25)
cv2.imwrite("lena_nearest_scratch_downscaled.png", downscaled_nn_scratch)
lena_nearest_scratch = nn_interpolation(downscaled_nn_scratch, 4)
cv2.imwrite("lena_nearest_scratch.png", lena_nearest_scratch)

# NEAREST NEIGHBOUR openCV
downscaled_nn = cv2.resize(lena, (w//4, h//4), interpolation=cv2.INTER_NEAREST)
cv2.imwrite("lena_nearest_cv_downscaled.png", downscaled_nn)
lena_nearest_cv = cv2.resize(downscaled_nn, (w, h), interpolation=cv2.INTER_NEAREST)
cv2.imwrite("lena_nearest_cv.png", lena_nearest_cv)

# BILINEAR scratch
def bl_interpolation(img, scale):
    h, w = img.shape[:2]
    new_h, new_w = int(h*scale), int(w*scale)
    resized_img =np.zeros((new_h, new_w), dtype=img.dtype)
    for i in range(new_h):
        for j in range(new_w):
            x = i / scale
            y = j / scale

            x0 = int(x)
            x1 = min(x0 + 1, h - 1)
            y0 = int(y)
            y1 = min(y0 + 1, w - 1)
            
            intensity_a = img[x0, y0]
            intensity_b = img[x0, y1]
            intensity_c = img[x1, y0]
            intensity_d = img[x1, y1]

            weight_a = (x1 - x) * (y1 - y)
            weight_b = (x1 - x) * (y - y0)
            weight_c = (x - x0) * (y1 - y)
            weight_d = (x - x0) * (y - y0)
            
            resized_img[i, j] = intensity_a * weight_a + intensity_b * weight_b + intensity_c * weight_c + intensity_d * weight_d

    return resized_img

downscaled_bl_scratch = bl_interpolation(lena, 0.25)
cv2.imwrite("lena_bilinear_scratch_downscaled.png", downscaled_bl_scratch)
lena_bilinear_scratch = bl_interpolation(downscaled_bl_scratch, 4)
cv2.imwrite("lena_bilinear_scratch.png", lena_bilinear_scratch)

# BILINEAR openCV
downscaled_bl = cv2.resize(lena, (w//4, h//4), interpolation = cv2.INTER_LINEAR)
cv2.imwrite("lena_bilinear_cv_downscaled.png", downscaled_bl)
lena_bilinear_cv = cv2.resize(downscaled_bl, (w, h), interpolation = cv2.INTER_LINEAR)
cv2.imwrite("lena_bilinear_cv.png", lena_bilinear_cv)

# BICUBIC openCV
downscaled_bc = cv2.resize(lena, (w//4, h//4), interpolation = cv2.INTER_CUBIC)
cv2.imwrite("lena_bicubic_cv_downscaled.png", downscaled_bc)
lena_bicubic_cv = cv2.resize(downscaled_bc, (w, h), interpolation = cv2.INTER_CUBIC)
cv2.imwrite("lena_bicubic_cv.png", lena_bicubic_cv)
