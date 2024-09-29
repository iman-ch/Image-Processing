import cv2
import numpy as np

def mse(img1, img2):
    assert img1.shape == img2.shape, "images need to have same dimensions"
    error = np.sum((img1 - img2) ** 2)
    mse_value = error / (img1.shape[0] * img1.shape[1])
    return mse_value

# LOAD IMAGES
lena = cv2.imread("C:\\Users\\imanc\\Documents\\FALL24\\CP467-ImageProccesing\\A1\\A1_chau0820\\Images\\lena.tif", 0)
lena_nearest_scratch = cv2.imread("C:\\Users\\imanc\\Documents\\FALL24\\CP467-ImageProccesing\\A1\\A1_chau0820\\lena_nearest_scratch.png", 0)
lena_nearest_cv = cv2.imread("C:\\Users\\imanc\\Documents\\FALL24\\CP467-ImageProccesing\\A1\\A1_chau0820\\lena_nearest_cv.png", 0)
lena_bilinear_scratch = cv2.imread("C:\\Users\\imanc\\Documents\\FALL24\\CP467-ImageProccesing\\A1\\A1_chau0820\\lena_bilinear_scratch.png", 0)
lena_bilinear_cv = cv2.imread("C:\\Users\\imanc\\Documents\\FALL24\\CP467-ImageProccesing\\A1\\A1_chau0820\\lena_bilinear_cv.png", 0)
lena_bicubic_cv = cv2.imread("C:\\Users\\imanc\\Documents\\FALL24\\CP467-ImageProccesing\\A1\\A1_chau0820\\lena_bicubic_cv.png", 0)

# MSE CALC
mse_nearest_scratch = mse(lena, lena_nearest_scratch)
mse_nearest_cv = mse(lena, lena_nearest_cv)
mse_bilinear_scratch = mse(lena, lena_bilinear_scratch)
mse_bilinear_cv = mse(lena, lena_bilinear_cv)
mse_bicubic_cv = mse(lena, lena_bicubic_cv)

# RESULTS
print(f"MSE (Nearest Neighbor Scratch): {mse_nearest_scratch}")
print(f"MSE (Nearest Neighbor OpenCV): {mse_nearest_cv}")
print(f"MSE (Bilinear Scratch): {mse_bilinear_scratch}")
print(f"MSE (Bilinear OpenCV): {mse_bilinear_cv}")
print(f"MSE (Bicubic OpenCV): {mse_bicubic_cv}")