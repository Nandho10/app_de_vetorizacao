import cv2
import numpy as np

p1 = r"C:\Users\Luiz.araujo\.gemini\antigravity\brain\4c3ff5ce-0a75-46ed-ace2-9ba571966979\.user_uploaded\media_1791227086305.png"
p2 = r"C:\Users\Luiz.araujo\.gemini\antigravity\brain\4c3ff5ce-0a75-46ed-ace2-9ba571966979\.user_uploaded\media_1791227142467.png"

img1 = cv2.imread(p1)
img2 = cv2.imread(p2)

print("Img 1 shape:", img1.shape if img1 is not None else None)
print("Img 2 shape:", img2.shape if img2 is not None else None)
