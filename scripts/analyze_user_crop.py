import cv2
import numpy as np

img = cv2.imread(r"D:\Documentos\Luiz\Antigravity\Vetorizador de quadras\Exemplos de quadras\44463-33-38.png")
p1 = r"C:\Users\Luiz.araujo\.gemini\antigravity\brain\4c3ff5ce-0a75-46ed-ace2-9ba571966979\.user_uploaded\media_1791227086305.png"
user_crop = cv2.imread(p1)

print("Original sheet shape:", img.shape)
print("User crop shape:", user_crop.shape)

# Let's inspect the green lines in user_crop:
# Green lines in user_crop have color like [0, 220, 50] or [22, 163, 74]
# Notice user_crop is a screenshot of the Leaflet web map or QGIS viewer!
# In the user screenshot, there are green polygon boundaries and black raster pixels from 44463.
