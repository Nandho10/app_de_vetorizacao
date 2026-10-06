import cv2
import numpy as np

img = cv2.imread(r"D:\Documentos\Luiz\Antigravity\Vetorizador de quadras\Exemplos de quadras\44463-33-38.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.bilateralFilter(gray, 5, 50, 50)
bin_inv = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10)

# Let's inspect where in 44463 the user crop was taken:
# In the user crop, we have "7.00" above a horizontal line and a vertical line going down.
# Let's find "7.00" in 44463 using template matching with a crop from media_1791227086305.png!
user_img = cv2.imread(r"C:\Users\Luiz.araujo\.gemini\antigravity\brain\4c3ff5ce-0a75-46ed-ace2-9ba571966979\.user_uploaded\media_1791227086305.png")

# Let's find matches or find coordinates
print("User img size:", user_img.shape)
