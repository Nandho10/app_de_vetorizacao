import cv2
import numpy as np
import scipy.ndimage as ndi
from shapely.geometry import Polygon
import geopandas as gpd
import matplotlib.pyplot as plt

img_path = r"D:\Documentos\Luiz\Antigravity\Vetorizador de quadras\Exemplos de quadras\44463-33-38.png"
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h_img, w_img = gray.shape
img_area = h_img * w_img

bin_inv = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY_INV, 25, 10)

contours, _ = cv2.findContours(bin_inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
quadra_cnt = max([c for c in contours if 0.08 * img_area < cv2.contourArea(c) < 0.6 * img_area], key=cv2.contourArea)

quadra_mask = np.zeros((h_img, w_img), dtype=np.uint8)
cv2.drawContours(quadra_mask, [quadra_cnt], -1, 255, thickness=cv2.FILLED)

# MORPHOLOGICAL OPENING ON QUADRA MASK TO CUT OFF ARROWS/SPURS LIKE "P.R."
kernel_quadra = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
quadra_mask = cv2.morphologyEx(quadra_mask, cv2.MORPH_OPEN, kernel_quadra)
quadra_mask = cv2.morphologyEx(quadra_mask, cv2.MORPH_CLOSE, kernel_quadra)

# Re-extract clean quadra contour
q_cnts, _ = cv2.findContours(quadra_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
quadra_cnt = max(q_cnts, key=cv2.contourArea)
arc = cv2.arcLength(quadra_cnt, True)
approx_quadra = cv2.approxPolyDP(quadra_cnt, 0.002 * arc, True)

# Update quadra_mask with clean smoothed boundary
quadra_mask = np.zeros((h_img, w_img), dtype=np.uint8)
cv2.drawContours(quadra_mask, [approx_quadra], -1, 255, thickness=cv2.FILLED)

inside_lines = cv2.bitwise_and(bin_inv, bin_inv, mask=quadra_mask)
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inside_lines, connectivity=8)

clean_lines = np.zeros_like(bin_inv)
cv2.drawContours(clean_lines, [approx_quadra], -1, 255, thickness=3)

for label in range(1, num_labels):
    w = stats[label, cv2.CC_STAT_WIDTH]
    h = stats[label, cv2.CC_STAT_HEIGHT]
    area = stats[label, cv2.CC_STAT_AREA]
    diag = np.hypot(w, h)
    if diag > 50 and (w > 25 or h > 25) and area > 80:
        clean_lines[labels == label] = 255

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
closed_lines = cv2.dilate(clean_lines, kernel, iterations=1)
cv2.drawContours(closed_lines, [approx_quadra], -1, 255, thickness=4)

lots_binary = cv2.bitwise_and(cv2.bitwise_not(closed_lines), quadra_mask)
kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
lots_binary = cv2.morphologyEx(lots_binary, cv2.MORPH_OPEN, kernel_erode)

num_lots, lot_labels = cv2.connectedComponents(lots_binary, connectivity=8)
quadra_area_px = cv2.contourArea(quadra_cnt)

filtered_labels = np.zeros_like(lot_labels)
current_id = 1
for l in range(1, num_lots):
    area = np.sum(lot_labels == l)
    if 0.002 * quadra_area_px < area < 0.25 * quadra_area_px:
        filtered_labels[lot_labels == l] = current_id
        current_id += 1

# ZERO GAP: Distance Transform
_, (row_idx, col_idx) = ndi.distance_transform_edt(filtered_labels == 0, return_indices=True)
full_partition = filtered_labels[row_idx, col_idx]
full_partition[quadra_mask == 0] = 0

meters_per_pixel = (0.0254 / 300) * 750
bx, by, bw, bh = cv2.boundingRect(approx_quadra)

records = []
for lid in range(1, current_id):
    lot_mask = (full_partition == lid).astype(np.uint8) * 255
    # Smooth lot mask slightly to eliminate single-pixel notches
    lot_mask = cv2.morphologyEx(lot_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    lot_mask = ndi.binary_fill_holes(lot_mask).astype(np.uint8) * 255
    
    cnts, _ = cv2.findContours(lot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cnt = max(cnts, key=cv2.contourArea)
        epsilon = 0.003 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        coords = []
        for pt in approx:
            px, py = pt[0]
            mx = (px - bx) * meters_per_pixel
            my = (bh - (py - by)) * meters_per_pixel
            coords.append((round(mx, 3), round(my, 3)))
        if coords[0] != coords[-1]:
            coords.append(coords[0])
            
        p = Polygon(coords)
        if not p.is_valid:
            p = p.buffer(0)
        if p.geom_type == 'Polygon' and len(p.interiors) > 0:
            p = Polygon(p.exterior.coords)
            
        records.append({
            "NUM_LOTE": f"{lid:02d}",
            "AREA_M2": round(p.area, 2),
            "PERIM_M": round(p.length, 2),
            "geometry": p
        })

gdf = gpd.GeoDataFrame(records, crs="EPSG:31983")
fig, ax = plt.subplots(figsize=(10, 14))
gdf.plot(ax=ax, color='#e27d80', edgecolor='#222222', linewidth=0.9)
for idx, row in gdf.iterrows():
    pt = row.geometry.representative_point()
    ax.annotate(text=row['NUM_LOTE'], xy=(pt.x, pt.y),
                horizontalalignment='center', verticalalignment='center',
                fontsize=7.5, color='black', weight='bold')

plt.title(f"Lotes Sem Espaço e Sem Esporos ({len(gdf)} lotes)", fontsize=13)
plt.axis('equal')
plt.tight_layout()
plt.savefig(r"D:\Documentos\Luiz\Antigravity\Vetorizador de quadras\zero_gap_clean.png", dpi=200)
print("Saved zero_gap_clean.png, lot count:", len(gdf))
