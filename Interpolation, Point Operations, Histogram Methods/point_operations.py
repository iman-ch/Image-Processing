import cv2
import numpy as np

# LOAD CAMERAMAN IMAGE
cameraman = cv2.imread('Images/cameraman.tif', cv2.IMREAD_GRAYSCALE)

# NEGATIVE 
def negative(img):
    neg_image = 256 - 1 - img
    return neg_image

cameraman_negative = negative(cameraman)
cv2.imwrite("cameraman_negative.png", cameraman_negative)

# POWER-LAW / GAMMA
def power_law(img, gamma):
    corrected = np.array(255*(img / 255) ** gamma, dtype = 'uint8')
    return corrected

cameraman_power = power_law(cameraman, 0.65)
cv2.imwrite(f"cameraman_power.png", cameraman_power)

# BIT-PLANE SLICING
def bit_plane_slicing(img, bit):
    bit_image = ((img >> bit) & 1) * 255
    return bit_image

for bit in range(8):
    cameraman_bit = bit_plane_slicing(cameraman, bit)
    cv2.imwrite(f"cameraman_b{bit+1}.png", cameraman_bit)