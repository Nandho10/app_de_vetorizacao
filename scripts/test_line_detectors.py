import cv2
import numpy as np

# Load 44463
img = cv2.imread(r"D:\Documentos\Luiz\Antigravity\Vetorizador de quadras\Exemplos de quadras\44463-33-38.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.bilateralFilter(gray, 5, 50, 50)
bin_inv = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10)

# LSD (Line Segment Detector)
lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
lines, width, prec, nfa = lsd.detect(gray)
print("LSD lines detected:", len(lines) if lines is not None else 0)

# HoughLinesP
edges = cv2.Canny(blurred, 50, 150)
hough = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=40, minLineLength=30, maxLineGap=10)
print("Hough lines detected:", len(hough) if hough is not None else 0)
