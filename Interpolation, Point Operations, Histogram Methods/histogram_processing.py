import cv2
import numpy as np
import matplotlib.pyplot as plt

# LOAD IMAGES
einstein = cv2.imread("C:\\Users\\imanc\\Documents\\FALL24\\CP467-ImageProccesing\\A1\\A1_chau0820\\Images\\einstein.tif", 0)
chest_xray1 = cv2.imread("C:\\Users\\imanc\\Documents\\FALL24\\CP467-ImageProccesing\\A1\\A1_chau0820\\Images\\chest_x-ray1.jpeg", 0)
chest_xray2 = cv2.imread("C:\\Users\\imanc\\Documents\\FALL24\\CP467-ImageProccesing\\A1\\A1_chau0820\\Images\\chest_x-ray2.jpeg", 0)

# HISTOGRAM EQUILIZATION
einstein_array = np.asarray(einstein)
flat = einstein_array.flatten()

def get_histogram(img, bins):
    histogram = np.zeros(bins)
    for pixel in img:
        histogram[pixel] += 1
    return histogram

def cumsum(a):
    a = iter(a)
    b = [next(a)]
    for i in a:
        b.append(b[-1] + i)
    return np.array(b)

hist = get_histogram(flat, 256)
cs = cumsum(hist)

nj = (cs - cs.min()) * 255
N = cs.max() - cs.min()
cs = nj / N
cs = cs.astype('uint8')

einstein_equalized = np.reshape(cs[flat], einstein.shape)
cv2.imwrite('einstein_equalized.png', einstein_equalized)

# HISTOGRAM MATCHING
def hist_match(source, template):
    old = source.shape
    source = source.ravel()
    template = template.ravel()

    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(old)

chest_xray3 = hist_match(chest_xray1, chest_xray2)
cv2.imwrite('chest_x-ray3.png', chest_xray3)