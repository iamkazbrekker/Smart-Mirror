import numpy as np
import cv2 as cv

def apply_rgb(roi):
    return cv.cvtColor(roi, cv.COLOR_BGR2RGB)

def apply_gray(roi):
    return cv.cvtColor(roi, cv.COLOR_BGR2GRAY)

def apply_hsv(roi):
    return cv.cvtColor(roi, cv.COLOR_BGR2HSV)

def apply_blur(roi):
    return cv.GaussianBlur(roi, (15,15), 0)

def apply_edge(roi):
    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 100, 200)
    return cv.cvtColor(edges, cv.COLOR_GRAY2BGR)

def apply_sepia(roi):
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    sepia = cv.transform(roi, kernel)
    return np.clip(sepia, 0, 255).astype(np.uint8)

def apply_invert(roi):
    return cv.bitwise_not(roi)

def apply_warm(roi):
    incr_ch_lut = np.array([min(255, i + 40) for i in range(256)]).astype("uint8")
    decr_ch_lut = np.array([max(0, i - 40) for i in range(256)]).astype("uint8")
    c_b, c_g, c_r = cv.split(roi)
    c_r = cv.LUT(c_r, incr_ch_lut)
    c_b = cv.LUT(c_b, decr_ch_lut)
    return cv.merge((c_b, c_g, c_r))

def apply_cool(roi):
    incr_ch_lut = np.array([min(255, i + 40) for i in range(256)]).astype("uint8")
    decr_ch_lut = np.array([max(0, i - 40) for i in range(256)]).astype("uint8")
    c_b, c_g, c_r = cv.split(roi)
    c_r = cv.LUT(c_r, decr_ch_lut)
    c_b = cv.LUT(c_b, incr_ch_lut)
    return cv.merge((c_b, c_g, c_r))

def apply_cartoon(roi):
    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray, 7)
    edges = cv.adaptiveThreshold(gray, 255,
                                 cv.ADAPTIVE_THRESH_MEAN_C,
                                 cv.THRESH_BINARY, 9, 9)
    color = cv.bilateralFilter(roi, 9, 250, 250)
    cartoon = cv.bitwise_and(color, color, mask=edges)
    return cartoon
