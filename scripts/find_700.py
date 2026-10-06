import cv2
import numpy as np

img = cv2.imread(r"D:\Documentos\Luiz\Antigravity\Vetorizador de quadras\Exemplos de quadras\44463-33-38.png")
# Let's inspect where y is near the bottom lots or top lots
# In 44463, lots are arranged along the top (y ~ 500..800) and bottom (y ~ 1400..1700).
# Front of bottom lots is along the street at y ~ 1680
# Front of top lots is along the street at y ~ 540
# The central dividing line is at y ~ 1080 (fundos dos lotes)!
# The image shows "7.00" right at the lot boundary line!
print("44463 loaded successfully.")
