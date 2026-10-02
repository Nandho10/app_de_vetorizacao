import os
import cv2
import numpy as np
import scipy.ndimage as ndi
from shapely.geometry import Polygon
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Paths
img_path = r"D:\Documentos\Luiz\Antigravity\Vetorizador de quadras\Exemplos de quadras\44463-33-38.png"
art_dir = r"C:\Users\Luiz.araujo\.gemini\antigravity\brain\4c3ff5ce-0a75-46ed-ace2-9ba571966979"
os.makedirs(art_dir, exist_ok=True)

img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h_img, w_img = gray.shape
img_area = h_img * w_img

bin_inv = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY_INV, 25, 10)

contours, _ = cv2.findContours(bin_inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
quadra_cnt_raw = max([c for c in contours if 0.04 * img_area < cv2.contourArea(c) < 0.7 * img_area], key=cv2.contourArea)

# 1. Pipeline Original (Com Gaps e com esporo)
quadra_mask_raw = np.zeros((h_img, w_img), dtype=np.uint8)
cv2.drawContours(quadra_mask_raw, [quadra_cnt_raw], -1, 255, thickness=cv2.FILLED)

inside_raw = cv2.bitwise_and(bin_inv, bin_inv, mask=quadra_mask_raw)
num_l, lbls, stats, _ = cv2.connectedComponentsWithStats(inside_raw, connectivity=8)
clean_lines_raw = np.zeros_like(bin_inv)
cv2.drawContours(clean_lines_raw, [quadra_cnt_raw], -1, 255, thickness=3)
for l in range(1, num_l):
    if np.hypot(stats[l, 2], stats[l, 3]) > 40 and stats[l, 4] > 60:
        clean_lines_raw[lbls == l] = 255
closed_raw = cv2.dilate(clean_lines_raw, np.ones((3,3), np.uint8), iterations=1)
lots_bin_raw = cv2.bitwise_and(cv2.bitwise_not(closed_raw), quadra_mask_raw)
lots_bin_raw = cv2.morphologyEx(lots_bin_raw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))
cnts_raw, _ = cv2.findContours(lots_bin_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

q_area = cv2.contourArea(quadra_cnt_raw)
meters_per_pixel = (0.0254 / 300) * 750
bx, by, bw, bh = cv2.boundingRect(quadra_cnt_raw)

gaps_records = []
for c in cnts_raw:
    if 0.002 * q_area < cv2.contourArea(c) < 0.35 * q_area:
        ap = cv2.approxPolyDP(c, 0.004 * cv2.arcLength(c, True), True)
        if len(ap) >= 3:
            coords = [((pt[0][0] - bx) * meters_per_pixel, (bh - (pt[0][1] - by)) * meters_per_pixel) for pt in ap]
            coords.append(coords[0])
            p = Polygon(coords)
            if p.is_valid and p.area > 10:
                gaps_records.append({"geometry": p})
gdf_gaps = gpd.GeoDataFrame(gaps_records, crs="EPSG:31983")

# 2. Pipeline Zero-Gap Atualizado
kernel_q = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
quadra_mask_clean = cv2.morphologyEx(quadra_mask_raw, cv2.MORPH_OPEN, kernel_q)
quadra_mask_clean = cv2.morphologyEx(quadra_mask_clean, cv2.MORPH_CLOSE, kernel_q)
q_cnts, _ = cv2.findContours(quadra_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
quadra_cnt_clean = max(q_cnts, key=cv2.contourArea)
approx_q = cv2.approxPolyDP(quadra_cnt_clean, 0.002 * cv2.arcLength(quadra_cnt_clean, True), True)
quadra_mask_clean = np.zeros((h_img, w_img), dtype=np.uint8)
cv2.drawContours(quadra_mask_clean, [approx_q], -1, 255, thickness=cv2.FILLED)

inside_lines = cv2.bitwise_and(bin_inv, bin_inv, mask=quadra_mask_clean)
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inside_lines, connectivity=8)
clean_lines = np.zeros_like(bin_inv)
cv2.drawContours(clean_lines, [approx_q], -1, 255, thickness=3)
for label in range(1, num_labels):
    if np.hypot(stats[label, 2], stats[label, 3]) > 50 and stats[label, 4] > 80:
        clean_lines[labels == label] = 255
closed_lines = cv2.dilate(clean_lines, np.ones((3,3), np.uint8), iterations=1)
cv2.drawContours(closed_lines, [approx_q], -1, 255, thickness=4)
lots_binary = cv2.bitwise_and(cv2.bitwise_not(closed_lines), quadra_mask_clean)
lots_binary = cv2.morphologyEx(lots_binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))

num_lots, lot_labels = cv2.connectedComponents(lots_binary, connectivity=8)
q_area_clean = cv2.contourArea(quadra_cnt_clean)
filtered_labels = np.zeros_like(lot_labels)
curr_id = 1
for l in range(1, num_lots):
    if 0.0015 * q_area_clean < np.sum(lot_labels == l) < 0.35 * q_area_clean:
        filtered_labels[lot_labels == l] = curr_id
        curr_id += 1

_, (r_idx, c_idx) = ndi.distance_transform_edt(filtered_labels == 0, return_indices=True)
partition = filtered_labels[r_idx, c_idx]
partition[quadra_mask_clean == 0] = 0

zero_records = []
kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
for lid in range(1, curr_id):
    lm = (partition == lid).astype(np.uint8) * 255
    lm = cv2.morphologyEx(lm, cv2.MORPH_CLOSE, kernel_close)
    lm = ndi.binary_fill_holes(lm).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(lm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cnt = max(cnts, key=cv2.contourArea)
        ap = cv2.approxPolyDP(cnt, max(1.0, 0.0035 * cv2.arcLength(cnt, True)), True)
        coords = [((pt[0][0] - bx) * meters_per_pixel, (bh - (pt[0][1] - by)) * meters_per_pixel) for pt in ap]
        coords.append(coords[0])
        p = Polygon(coords)
        if not p.is_valid:
            p = p.buffer(0)
        if p.geom_type == 'Polygon' and len(p.interiors) > 0:
            p = Polygon(p.exterior.coords)
        zero_records.append({
            "NUM_LOTE": f"{lid:02d}",
            "AREA_M2": round(p.area, 2),
            "PERIM_M": round(p.length, 2),
            "geometry": p
        })

gdf_zero = gpd.GeoDataFrame(zero_records, crs="EPSG:31983")

# PLOT COMPARATIVO LADO A LADO
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 11), facecolor='#0f172a')

# Subplot 1: Com Gaps (Anterior)
ax1.set_facecolor('#1e293b')
gdf_gaps.plot(ax=ax1, color='#ef4444', edgecolor='#fca5a5', alpha=0.75, linewidth=1.2)
ax1.set_title("Vetorização Anterior: COM ESPAÇOS (Corredores vazios)", color='#f87171', fontsize=14, pad=15, weight='bold')
ax1.set_xlabel("Coordenadas Cartesianas (Metros)", color='#94a3b8')
ax1.set_ylabel("Coordenadas Cartesianas (Metros)", color='#94a3b8')
ax1.tick_params(colors='#94a3b8')
ax1.grid(color='#334155', linestyle='--', linewidth=0.5, alpha=0.7)
ax1.axis('equal')

# Subplot 2: Zero Gap (Ajustado)
ax2.set_facecolor('#1e293b')
gdf_zero.plot(ax=ax2, color='#10b981', edgecolor='#064e3b', alpha=0.85, linewidth=1.0)
for idx, row in gdf_zero.iterrows():
    pt = row.geometry.representative_point()
    ax2.annotate(text=row['NUM_LOTE'], xy=(pt.x, pt.y),
                horizontalalignment='center', verticalalignment='center',
                fontsize=7.5, color='#ffffff', weight='bold')

ax2.set_title(f"Vetorização Nova: ZERO GAP (43 Lotes Contíguos)", color='#34d399', fontsize=14, pad=15, weight='bold')
ax2.set_xlabel("Coordenadas Cartesianas (Metros)", color='#94a3b8')
ax2.set_ylabel("Coordenadas Cartesianas (Metros)", color='#94a3b8')
ax2.tick_params(colors='#94a3b8')
ax2.grid(color='#334155', linestyle='--', linewidth=0.5, alpha=0.7)
ax2.axis('equal')

plt.suptitle("Comparativo Topológico Cadastral - Planta de Quadra Fiscal (Escala 1:750 - Metros Reais)", 
             color='#f1f5f9', fontsize=16, weight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])

out_plot_art = os.path.join(art_dir, "comparativo_zero_gap.png")
out_plot_ws = r"D:\Documentos\Luiz\Antigravity\Vetorizador de quadras\comparativo_zero_gap.png"
plt.savefig(out_plot_art, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
plt.savefig(out_plot_ws, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
plt.close()

# GERAR TAMBÉM O PLOT DETALHADO DO RESULTADO FINAL ISOLADO
fig2, ax = plt.subplots(figsize=(14, 10), facecolor='#0f172a')
ax.set_facecolor('#1e293b')

# Gradiente de cores agradável para os lotes
colors = plt.cm.tab20(np.linspace(0, 1, len(gdf_zero)))
gdf_zero.plot(ax=ax, color=colors, edgecolor='#0f172a', linewidth=1.5, alpha=0.85)

for idx, row in gdf_zero.iterrows():
    pt = row.geometry.representative_point()
    ax.annotate(text=f"{row['NUM_LOTE']}\n{int(row['AREA_M2'])}m²", xy=(pt.x, pt.y),
                horizontalalignment='center', verticalalignment='center',
                fontsize=7, color='#ffffff', weight='bold',
                bbox=dict(boxstyle="round,pad=0.15", fc="#0f172a", ec="none", alpha=0.5))

ax.set_title(f"Planta de Quadra Fiscal Vetorizada - Zero Gap (Total: {len(gdf_zero)} Lotes | SIRGAS 2000 / UTM 23S)", 
             color='#f8fafc', fontsize=15, pad=18, weight='bold')
ax.set_xlabel("Distância X (metros)", color='#94a3b8', fontsize=11)
ax.set_ylabel("Distância Y (metros)", color='#94a3b8', fontsize=11)
ax.tick_params(colors='#94a3b8')
ax.grid(color='#334155', linestyle='--', linewidth=0.6, alpha=0.6)
ax.axis('equal')
plt.tight_layout()

out_detail_art = os.path.join(art_dir, "plot_zero_gap_detalhado.png")
out_detail_ws = r"D:\Documentos\Luiz\Antigravity\Vetorizador de quadras\plot_zero_gap_detalhado.png"
plt.savefig(out_detail_art, dpi=220, facecolor=fig2.get_facecolor(), edgecolor='none')
plt.savefig(out_detail_ws, dpi=220, facecolor=fig2.get_facecolor(), edgecolor='none')
plt.close()

print(f"Salvo comparativo em: {out_plot_art}")
print(f"Salvo detalhado em: {out_detail_art}")
