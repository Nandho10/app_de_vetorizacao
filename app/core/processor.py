import os
import cv2
import numpy as np
import tempfile
import zipfile
from PIL import Image
from shapely.geometry import Polygon, MultiPolygon, Point, LineString, MultiLineString
from shapely.ops import polygonize, unary_union, nearest_points
import geopandas as gpd
import scipy.ndimage as ndi

try:
    import pypdfium2 as pdfium
except ImportError:
    pdfium = None


class QuadraProcessor:
    """
    Engine para leitura, extração de contornos e vetorização de plantas de quadra fiscais.
    Identifica o perímetro da quadra, segmenta lotes com topologia zero gap,
    detecta edificações existentes (amarelas) e edificações demolidas/áreas livres (vermelhas).
    """

    def __init__(self, scale_denom=750, dpi=300, simplify_tol=0.002, min_lot_area_m2=10.0, epsg=31983):
        self.scale_denom = float(scale_denom)
        self.dpi = float(dpi)
        self.simplify_tol = float(simplify_tol)
        self.min_lot_area_m2 = float(min_lot_area_m2)
        self.epsg = int(epsg) if epsg else 31983
        self.meters_per_pixel = (0.0254 / self.dpi) * self.scale_denom

    @staticmethod
    def load_image(file_path):
        """Carrega imagem de formatos .pdf, .tif, .jpg, .png."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            if pdfium is None:
                raise RuntimeError("Biblioteca pypdfium2 não instalada para suporte a PDF.")
            pdf = pdfium.PdfDocument(file_path)
            page = pdf[0]
            scale = 300 / 72.0
            pil_image = page.render(scale=scale).to_pil()
            img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            return img
        else:
            img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                pil_img = Image.open(file_path).convert("RGB")
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            return img

    def binarize(self, img, adaptive_block=25, adaptive_c=10):
        """
        Binariza a imagem capturando traçados cadastrais em preto, azul escuro ou roxo/violeta,
        e ignorando tons de amarelo e vermelho (edificações existentes e demolidas).
        """
        if len(img.shape) == 2 or img.shape[2] == 1:
            gray = img
            blurred = cv2.bilateralFilter(gray, 5, 50, 50)
            bin_inv = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, adaptive_block, adaptive_c
            )
            return gray, bin_inv

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred_gray = cv2.bilateralFilter(gray, 5, 50, 50)
        bin_gray = cv2.adaptiveThreshold(
            blurred_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, adaptive_block, adaptive_c
        )

        # Canal vermelho: máxima absorção/contraste para linhas azuis e pretas
        red_ch = img[:, :, 2]
        blurred_red = cv2.bilateralFilter(red_ch, 5, 50, 50)
        bin_red = cv2.adaptiveThreshold(
            blurred_red, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, adaptive_block, adaptive_c
        )

        # Combina canais para capturar traçados pretos, azuis e roxos
        bin_inv = cv2.bitwise_or(bin_gray, bin_red)
        return gray, bin_inv

    def find_quadra_boundary(self, bin_inv, crop_bbox=None):
        """
        Localiza o contorno principal da quadra, fechando pequenas descontinuidades de texto
        (como círculos de vértice ou cotas nos cantos de lotes como Lote 34) e eliminando
        esporos/setas externos (como ponteiro 'P.R.').
        """
        h_img, w_img = bin_inv.shape
        img_area = h_img * w_img

        if crop_bbox:
            bx, by, bw, bh = crop_bbox
            mask = np.zeros_like(bin_inv)
            mask[by:by+bh, bx:bx+bw] = 255
            bin_inv_crop = cv2.bitwise_and(bin_inv, mask)
            contours, _ = cv2.findContours(bin_inv_crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        else:
            # Fechamento morfológico para conectar quebras tênues no perímetro externo
            kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            bin_inv_closed = cv2.morphologyEx(bin_inv, cv2.MORPH_CLOSE, kernel_close)
            contours, _ = cv2.findContours(bin_inv_closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 0.015 * img_area < area < 0.70 * img_area:
                x, y, w, h = cv2.boundingRect(cnt)
                if w < 0.98 * w_img and h < 0.98 * h_img:
                    candidates.append((area, cnt))

        if not candidates:
            candidates = [(cv2.contourArea(c), c) for c in contours if cv2.contourArea(c) < 0.95 * img_area]

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0], reverse=True)
        best_cnt = candidates[0][1]

        # Remoção morfológica de setas/esporos finos externos
        quadra_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        cv2.drawContours(quadra_mask, [best_cnt], -1, 255, thickness=cv2.FILLED)
        kernel_quadra = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        quadra_mask = cv2.morphologyEx(quadra_mask, cv2.MORPH_OPEN, kernel_quadra)
        quadra_mask = cv2.morphologyEx(quadra_mask, cv2.MORPH_CLOSE, kernel_quadra)

        q_cnts, _ = cv2.findContours(quadra_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if q_cnts:
            best_cnt = max(q_cnts, key=cv2.contourArea)

        arc = cv2.arcLength(best_cnt, True)
        approx = cv2.approxPolyDP(best_cnt, 0.002 * arc, True)
        approx = self.clean_contour_spurs(approx)
        return approx

    @staticmethod
    def clean_contour_spurs(cnt, max_base_dist=35, min_height=15):
        """
        Remove espigões/esporos agudos (como setas P.R., ponteiros de cota)
        que saem do perímetro e retornam quase ao mesmo ponto.
        """
        pts = cnt.reshape(-1, 2)
        n = len(pts)
        if n < 6:
            return cnt
        cleaned = []
        i = 0
        while i < n:
            p_curr = pts[i]
            removed_spike = False
            for jump in [2, 3]:
                next_idx = (i + jump) % n
                p_next = pts[next_idx]
                base_dist = float(np.hypot(*(p_next - p_curr)))
                if base_dist <= max_base_dist:
                    for mid_offset in range(1, jump):
                        p_mid = pts[(i + mid_offset) % n]
                        line_vec = p_next - p_curr
                        if base_dist > 0:
                            dist_to_line = abs(float(np.cross(line_vec, p_curr - p_mid))) / base_dist
                        else:
                            dist_to_line = float(np.hypot(*(p_mid - p_curr)))
                        if dist_to_line >= min_height:
                            cleaned.append(p_curr)
                            i = next_idx
                            removed_spike = True
                            break
                if removed_spike:
                    break
            if not removed_spike:
                cleaned.append(p_curr)
                i += 1
        return np.array(cleaned, dtype=np.int32).reshape((-1, 1, 2))

    def get_building_suppression_mask(self, img, quadra_cnt):
        """
        Identifica áreas de edificações (tons de amarelo - existentes, e tons de vermelho/rosa - demolidas)
        para ignorá-las e garantir que suas paredes internas não fragmentem os lotes cadastrais.
        """
        if len(img.shape) == 2 or img.shape[2] == 1:
            return np.zeros(img.shape[:2], dtype=np.uint8)

        h_img, w_img = img.shape[:2]
        quadra_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        cv2.drawContours(quadra_mask, [quadra_cnt], -1, 255, thickness=cv2.FILLED)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Amarelo: saturação alta (>55) e matiz em [18, 55]
        mask_yellow = cv2.inRange(hsv, (18, 55, 75), (55, 255, 255))
        
        # Vermelho/Rosa: matiz em [0, 18] ou [160, 180], saturação > 40
        mask_red1 = cv2.inRange(hsv, (0, 40, 75), (18, 255, 255))
        mask_red2 = cv2.inRange(hsv, (160, 40, 75), (180, 255, 255))
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

        # Restringir à área interna da quadra
        mask_yellow = cv2.bitwise_and(mask_yellow, quadra_mask)
        mask_red = cv2.bitwise_and(mask_red, quadra_mask)

        k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, k)
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, k)
        mask_yellow = ndi.binary_fill_holes(mask_yellow).astype(np.uint8) * 255

        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, k)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, k)
        mask_red = ndi.binary_fill_holes(mask_red).astype(np.uint8) * 255

        clean_bldgs_mask = cv2.bitwise_or(mask_yellow, mask_red)
        return clean_bldgs_mask

    def extract_lots(self, bin_inv, quadra_cnt, gray_img=None, line_sensitivity=50, sort_mode="cadastral_ring", clean_bldgs_mask=None):
        """Segmenta os lotes no interior do perímetro da quadra eliminando espaços (zero gap) e unificando as edificações aos lotes."""
        h_img, w_img = bin_inv.shape
        quadra_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        cv2.drawContours(quadra_mask, [quadra_cnt], -1, 255, thickness=cv2.FILLED)

        inside_lines = cv2.bitwise_and(bin_inv, bin_inv, mask=quadra_mask)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inside_lines, connectivity=8)
        
        clean_lines = np.zeros_like(bin_inv)
        cv2.drawContours(clean_lines, [quadra_cnt], -1, 255, thickness=3)

        min_diag = max(30, int(line_sensitivity))
        for label in range(1, num_labels):
            w = stats[label, cv2.CC_STAT_WIDTH]
            h = stats[label, cv2.CC_STAT_HEIGHT]
            area = stats[label, cv2.CC_STAT_AREA]
            diag = np.hypot(w, h)
            if diag > min_diag and (w > 20 or h > 20) and area > 70:
                clean_lines[labels == label] = 255

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed_lines = cv2.dilate(clean_lines, kernel, iterations=1)
        cv2.drawContours(closed_lines, [quadra_cnt], -1, 255, thickness=4)

        lots_binary = cv2.bitwise_and(cv2.bitwise_not(closed_lines), quadra_mask)

        # Unificação de edificações ao corpo do lote:
        # Cria passagens verticais conectando os recuos frontais e fundos através das edificações
        # sem violar divisas laterais de lotes ou a linha central da quadra.
        if clean_bldgs_mask is not None and np.sum(clean_bldgs_mask > 0) > 0:
            bx, by, bw, bh = cv2.boundingRect(quadra_cnt)
            mid_y = by + bh / 2.0
            passage_mask = np.zeros_like(lots_binary)
            num_bldgs, labels_bldgs, stats_bldgs, centroids_bldgs = cv2.connectedComponentsWithStats(clean_bldgs_mask)

            for b in range(1, num_bldgs):
                by_b = stats_bldgs[b, cv2.CC_STAT_TOP]
                bh_b = stats_bldgs[b, cv2.CC_STAT_HEIGHT]
                cx_b = int(centroids_bldgs[b][0])
                cy_b = int(centroids_bldgs[b][1])
                is_top = (cy_b < mid_y)
                y_start = max(by + 10, by_b - 12)
                y_end = min(by + bh - 10, by_b + bh_b + 12)
                if is_top:
                    y_end = min(y_end, int(mid_y - 8))
                else:
                    y_start = max(y_start, int(mid_y + 8))
                cv2.line(passage_mask, (cx_b, y_start), (cx_b, y_end), 255, thickness=5)

            passage_mask = cv2.bitwise_or(passage_mask, clean_bldgs_mask)
            lots_binary = cv2.bitwise_or(lots_binary, passage_mask)
            lots_binary = cv2.bitwise_and(lots_binary, cv2.bitwise_not(closed_lines))
            lots_binary = cv2.bitwise_or(lots_binary, cv2.bitwise_and(passage_mask, quadra_mask))
            cv2.drawContours(lots_binary, [quadra_cnt], -1, 0, thickness=5)

        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        lots_binary = cv2.morphologyEx(lots_binary, cv2.MORPH_OPEN, kernel_erode)

        num_lots, lot_labels, stats_lots, _ = cv2.connectedComponentsWithStats(lots_binary, connectivity=8)
        quadra_area_px = cv2.contourArea(quadra_cnt)

        filtered_labels = np.zeros_like(lot_labels)
        current_id = 1
        for l in range(1, num_lots):
            x = stats_lots[l, cv2.CC_STAT_LEFT]
            y = stats_lots[l, cv2.CC_STAT_TOP]
            w_s = stats_lots[l, cv2.CC_STAT_WIDTH]
            h_s = stats_lots[l, cv2.CC_STAT_HEIGHT]
            area = stats_lots[l, cv2.CC_STAT_AREA]
            if 0.002 * quadra_area_px < area < 0.35 * quadra_area_px:
                sub_lbl = lot_labels[y:y+h_s, x:x+w_s]
                comp = (sub_lbl == l)
                filtered_labels[y:y+h_s, x:x+w_s][comp] = current_id
                current_id += 1

        if current_id <= 1:
            return [], np.zeros_like(bin_inv), {}

        # Tesselacao Voronoi via Distance Transform euclidiano (Zero Gap)
        _, (row_idx, col_idx) = ndi.distance_transform_edt(filtered_labels == 0, return_indices=True)
        full_partition = filtered_labels[row_idx, col_idx]
        full_partition[quadra_mask == 0] = 0

        valid_lots = []
        for lid in range(1, current_id):
            lot_mask = (full_partition == lid).astype(np.uint8) * 255
            cnts, _ = cv2.findContours(lot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                cnt = max(cnts, key=cv2.contourArea)
                # Epsilon uniforme de 1.0px elimina degraus mantendo retas perfeitas sem desvios em cotas
                approx_cnt = cv2.approxPolyDP(cnt, 1.0, True)
                if len(approx_cnt) >= 3:
                    valid_lots.append((lid, approx_cnt))

        # Ordenação espacial: Anel Cadastral (Padrão de Loteamento) ou Varredura Linear
        if sort_mode == "cadastral_ring":
            bx, by, bw, bh = cv2.boundingRect(quadra_cnt)
            center_y = by + bh / 2.0
            top = []
            right = []
            bottom = []
            for lid, c in valid_lots:
                M = cv2.moments(c)
                cx = M["m10"] / M["m00"] if M["m00"] > 0 else 0
                cy = M["m01"] / M["m00"] if M["m00"] > 0 else 0
                if cx > bx + 0.82 * bw:
                    if cy < by + 0.35 * bh:
                        top.append((cx, cy, lid, c))
                    else:
                        right.append((cx, cy, lid, c))
                elif cy < center_y:
                    top.append((cx, cy, lid, c))
                else:
                    bottom.append((cx, cy, lid, c))
            top.sort(key=lambda x: x[0])     # Topo: esquerda -> direita (1..N)
            right.sort(key=lambda x: x[1])   # Lateral: topo -> base
            bottom.sort(key=lambda x: -x[0]) # Base: direita -> esquerda
            ordered = top + right + bottom
            lot_cnts = [item[3] for item in ordered]
            lot_id_map = {item[2]: f"{idx:02d}" for idx, item in enumerate(ordered, start=1)}
        else:
            def get_centroid(item):
                c = item[1]
                M = cv2.moments(c)
                if M["m00"] > 0:
                    return (round(M["m01"] / M["m00"], -1), M["m10"] / M["m00"])
                return (0, 0)
            valid_lots.sort(key=get_centroid)
            lot_cnts = [item[1] for item in valid_lots]
            lot_id_map = {item[0]: f"{idx:02d}" for idx, item in enumerate(valid_lots, start=1)}

        return lot_cnts, full_partition, lot_id_map

    @staticmethod
    def merge_collinear_lines(lines_metric, max_angle_diff=6.0, max_dist=1.20):
        """
        Agrupa e unifica segmentos colineares contíguos (ex: espinha central,
        divisores de fundo de lote) gerando retas contínuas perfeitas de CAD.
        """
        if not lines_metric:
            return []

        segments = []
        for l in lines_metric:
            coords = np.array(l.coords)
            p1 = coords[0]
            p2 = coords[-1]
            vec = p2 - p1
            length = float(np.hypot(*vec))
            if length < 0.2:
                continue
            u = vec / length
            if u[0] < 0 or (abs(u[0]) < 1e-6 and u[1] < 0):
                u = -u
                p1, p2 = p2, p1
            angle = float(np.degrees(np.arctan2(u[1], u[0])) % 180)
            segments.append({
                "p1": p1, "p2": p2, "u": u, "length": length,
                "angle": angle, "mid": (p1 + p2) / 2.0, "pts": [p1, p2]
            })

        n = len(segments)
        parent = list(range(n))
        def find(i):
            if parent[i] == i: return i
            parent[i] = find(parent[i])
            return parent[i]
        def union(i, j):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_i] = root_j

        for i in range(n):
            s1 = segments[i]
            for j in range(i + 1, n):
                s2 = segments[j]
                diff_ang = abs(s1["angle"] - s2["angle"])
                if diff_ang > 90:
                    diff_ang = abs(180 - diff_ang)

                is_both_horiz = (s1["angle"] <= 15 or s1["angle"] >= 165) and (s2["angle"] <= 15 or s2["angle"] >= 165)
                is_both_vert = (75 <= s1["angle"] <= 105) and (75 <= s2["angle"] <= 105)
                angle_tol = 15.0 if (is_both_horiz or is_both_vert) else max_angle_diff
                if diff_ang > angle_tol:
                    continue

                d_mid = s2["mid"] - s1["mid"]
                normal = np.array([-s1["u"][1], s1["u"][0]])
                lat_dist = abs(float(np.dot(d_mid, normal)))
                if lat_dist > max_dist:
                    continue

                proj_s1_1 = float(np.dot(s1["p1"], s1["u"]))
                proj_s1_2 = float(np.dot(s1["p2"], s1["u"]))
                min_s1, max_s1 = min(proj_s1_1, proj_s1_2), max(proj_s1_1, proj_s1_2)

                proj_s2_1 = float(np.dot(s2["p1"], s1["u"]))
                proj_s2_2 = float(np.dot(s2["p2"], s1["u"]))
                min_s2, max_s2 = min(proj_s2_1, proj_s2_2), max(proj_s2_1, proj_s2_2)

                gap = max(0.0, max(min_s1, min_s2) - min(max_s1, max_s2))
                if gap <= 1.5:
                    union(i, j)

        groups = {}
        for i in range(n):
            root = find(i)
            groups.setdefault(root, []).append(segments[i])

        merged_lines = []
        for root, segs in groups.items():
            all_pts = []
            for s in segs:
                all_pts.extend(s["pts"])
            all_pts = np.array(all_pts)
            mean = np.mean(all_pts, axis=0)
            cov = np.cov(all_pts.T)
            if cov.ndim == 0 or len(all_pts) == 2:
                p1 = segs[0]["p1"]
                p2 = segs[0]["p2"]
                merged_lines.append(LineString([(round(p1[0], 3), round(p1[1], 3)), (round(p2[0], 3), round(p2[1], 3))]))
                continue

            eigvals, eigvecs = np.linalg.eigh(cov)
            v = eigvecs[:, -1]
            projs = (all_pts - mean) @ v
            p1 = mean + np.min(projs) * v
            p2 = mean + np.max(projs) * v
            merged_lines.append(LineString([(round(p1[0], 3), round(p1[1], 3)), (round(p2[0], 3), round(p2[1], 3))]))

        return merged_lines

    def extract_divisas(self, full_partition, quadra_cnt, origin_bbox, quadra_code, bairro, h_img):
        """
        Extrai o traçado completo das linhas (LineString) sem gerar polígonos individuais
        e sem linhas duplicadas paralelas (Zero Gap real).
        """
        bx, by, bw, bh = origin_bbox
        m_per_px = self.meters_per_pixel

        # 1. Perímetro da quadra em metros e coordenadas
        quadra_poly = self.to_metric_polygon(quadra_cnt, origin_bbox, buffer_snap=False)
        quadra_line = LineString(quadra_poly.exterior.coords)

        # 2. Extração das divisas internas a partir das interfaces do full_partition
        shift_r = full_partition[:, 1:]
        orig_r = full_partition[:, :-1]
        shift_d = full_partition[1:, :]
        orig_d = full_partition[:-1, :]

        pairs = set()
        for a, b in zip(orig_r.flat, shift_r.flat):
            if a != b and a > 0 and b > 0:
                pairs.add(tuple(sorted((int(a), int(b)))) )
        for a, b in zip(orig_d.flat, shift_d.flat):
            if a != b and a > 0 and b > 0:
                pairs.add(tuple(sorted((int(a), int(b)))) )

        raw_segments = []
        for i, j in sorted(pairs):
            mask_v = ((orig_r == i) & (shift_r == j)) | ((orig_r == j) & (shift_r == i))
            y_v, x_v = np.where(mask_v)
            pts_v = list(zip(x_v + 0.5, y_v.astype(float)))

            mask_h = ((orig_d == i) & (shift_d == j)) | ((orig_d == j) & (shift_d == i))
            y_h, x_h = np.where(mask_h)
            pts_h = list(zip(x_h.astype(float), y_h + 0.5))

            all_pts = np.array(pts_v + pts_h)
            if len(all_pts) < 6:
                continue

            mean = np.mean(all_pts, axis=0)
            cov = np.cov(all_pts.T)
            if cov.ndim == 0:
                continue
            eigvals, eigvecs = np.linalg.eigh(cov)
            v = eigvecs[:, -1]
            projs = (all_pts - mean) @ v
            p1 = mean + np.min(projs) * v
            p2 = mean + np.max(projs) * v

            length_px = np.hypot(*(p2 - p1))
            if length_px * m_per_px < 0.8:
                continue

            mx1 = (p1[0] - bx) * m_per_px
            my1 = (bh - (p1[1] - by)) * m_per_px
            mx2 = (p2[0] - bx) * m_per_px
            my2 = (bh - (p2[1] - by)) * m_per_px

            raw_segments.append(LineString([(round(mx1, 3), round(my1, 3)), (round(mx2, 3), round(my2, 3))]))

        # 3. Mesclar segmentos colineares contíguos (ex: espinha central e divisas contínuas)
        merged_lines = self.merge_collinear_lines(raw_segments, max_angle_diff=6.0, max_dist=1.20)

        # 4. Snap topológico:
        # A) Snap nos limites externos da quadra
        snapped_lines = []
        for l in merged_lines:
            coords = list(l.coords)
            p_s = Point(coords[0])
            p_e = Point(coords[-1])
            if quadra_line.distance(p_s) < 0.80:
                p_s = nearest_points(quadra_line, p_s)[0]
            if quadra_line.distance(p_e) < 0.80:
                p_e = nearest_points(quadra_line, p_e)[0]
            snapped_lines.append(LineString([(round(p_s.x, 3), round(p_s.y, 3)), (round(p_e.x, 3), round(p_e.y, 3))]))

        # B) Snap entre divisas internas adjacentes (junções em T perfeitas)
        final_internal = []
        for idx, l in enumerate(snapped_lines):
            coords = list(l.coords)
            p_s = Point(coords[0])
            p_e = Point(coords[-1])
            for other_idx, other_l in enumerate(snapped_lines):
                if idx == other_idx:
                    continue
                if quadra_line.distance(p_s) > 0.1 and other_l.distance(p_s) < 0.50:
                    p_s = nearest_points(other_l, p_s)[0]
                if quadra_line.distance(p_e) > 0.1 and other_l.distance(p_e) < 0.50:
                    p_e = nearest_points(other_l, p_e)[0]

            seg = LineString([(round(p_s.x, 3), round(p_s.y, 3)), (round(p_e.x, 3), round(p_e.y, 3))])
            if seg.length >= 2.5:
                final_internal.append(seg)

        # 5. Montar registros de linhas com atributos
        divisa_records = []
        divisa_features_pixel = []

        # Adicionar o perímetro da quadra como linha
        perim_len = round(quadra_line.length, 2)
        quadra_coords_px = []
        cv_pts_quadra = []
        for mx, my in quadra_line.coords:
            px = bx + mx / m_per_px
            py = by + bh - (my / m_per_px)
            quadra_coords_px.append([round(float(px), 1), round(float(h_img - py), 1)])
            cv_pts_quadra.append([int(round(px)), int(round(py))])

        divisa_records.append({
            "ID_LINHA": "LIN-001",
            "QUADRA": quadra_code,
            "BAIRRO": bairro,
            "TIPO": "PERIMETRO",
            "COMPR_M": perim_len,
            "geometry": quadra_line,
            "cv_pts": cv_pts_quadra
        })

        divisa_features_pixel.append({
            "type": "Feature",
            "properties": {
                "TIPO": "DIVISA",
                "ID": "LIN-001",
                "TIPO_DIVISA": "Perímetro da Quadra",
                "COMPR_M": perim_len,
                "COLOR": "#0284c7"
            },
            "geometry": {
                "type": "LineString",
                "coordinates": quadra_coords_px
            }
        })

        # Adicionar as divisas internas
        for idx, seg in enumerate(final_internal, start=2):
            id_linha = f"LIN-{idx:03d}"
            length_m = round(seg.length, 2)
            coords = list(seg.coords)
            dx = coords[-1][0] - coords[0][0]
            dy = coords[-1][1] - coords[0][1]
            angle = abs(np.arctan2(dy, dx) * 180 / np.pi)
            if angle > 90: angle = 180 - angle

            if angle < 20:
                tipo = "DIVISA_ESPINHA" if length_m > 40 else "DIVISA_HORIZONTAL"
                tipo_descr = "Espinha Central" if length_m > 40 else "Divisa de Lote"
            else:
                tipo = "DIVISA_VERTICAL"
                tipo_descr = "Divisa Lateral de Lote"

            coords_pixel = []
            cv_pts = []
            for mx, my in coords:
                px = bx + mx / m_per_px
                py = by + bh - (my / m_per_px)
                coords_pixel.append([round(float(px), 1), round(float(h_img - py), 1)])
                cv_pts.append([int(round(px)), int(round(py))])

            divisa_records.append({
                "ID_LINHA": id_linha,
                "QUADRA": quadra_code,
                "BAIRRO": bairro,
                "TIPO": tipo,
                "COMPR_M": length_m,
                "geometry": seg,
                "cv_pts": cv_pts
            })

            divisa_features_pixel.append({
                "type": "Feature",
                "properties": {
                    "TIPO": "DIVISA",
                    "ID": id_linha,
                    "TIPO_DIVISA": tipo_descr,
                    "COMPR_M": length_m,
                    "COLOR": "#ef4444"
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": coords_pixel
                }
            })

        if divisa_records:
            gdf_divisas = gpd.GeoDataFrame(divisa_records, crs=f"EPSG:{self.epsg}")
        else:
            gdf_divisas = gpd.GeoDataFrame(
                columns=["ID_LINHA", "QUADRA", "BAIRRO", "TIPO", "COMPR_M", "geometry"],
                crs=f"EPSG:{self.epsg}"
            )

        return gdf_divisas, divisa_features_pixel

    def to_metric_polygon(self, contour, origin_bbox, buffer_snap=True):
        """Converte coordenadas de pixels de imagem para coordenadas cartesianas em metros reais com zero gap."""
        bx, by, bw, bh = origin_bbox
        coords = []
        for pt in contour:
            px, py = pt[0]
            mx = (px - bx) * self.meters_per_pixel
            my = (bh - (py - by)) * self.meters_per_pixel
            coords.append((round(mx, 3), round(my, 3)))
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        
        poly = Polygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.geom_type == 'Polygon' and len(poly.interiors) > 0:
            poly = Polygon(poly.exterior.coords)
        if buffer_snap and poly.is_valid:
            poly = poly.buffer(0.5 * self.meters_per_pixel, join_style=2)
            if poly.geom_type == 'MultiPolygon':
                poly = max(poly.geoms, key=lambda g: g.area)
        return poly

    def to_pixel_polygon(self, contour, h_img):
        """Converte coordenadas de pixel para GeoJSON de overlay no visualizador (Y invertido para Leaflet L.CRS.Simple)."""
        coords = []
        for pt in contour:
            px, py = pt[0]
            coords.append([int(px), int(h_img - py)])
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        return coords

    def process(self, file_path, quadra_code="QD-01", bairro="", scale_override=None, line_sensitivity=50, crop_bbox=None, sort_mode="cadastral_ring", ref_cota_meters=None):
        if scale_override:
            self.scale_denom = float(scale_override)
            self.meters_per_pixel = (0.0254 / self.dpi) * self.scale_denom

        img = self.load_image(file_path)
        h_img, w_img = img.shape[:2]
        gray, bin_inv = self.binarize(img)
        
        quadra_cnt = self.find_quadra_boundary(bin_inv, crop_bbox=crop_bbox)
        if quadra_cnt is None:
            raise ValueError("Não foi possível identificar o perímetro da quadra automaticamente.")

        bx, by, bw, bh = cv2.boundingRect(quadra_cnt)
        origin_bbox = (bx, by, bw, bh)

        # 1. Máscara de edificações a ignorar (tons de amarelo e vermelho/rosa)
        clean_bldgs_mask = self.get_building_suppression_mask(img, quadra_cnt)

        # 2. Extrair lotes cadastrais unificando áreas internas para não fragmentar lotes
        lot_cnts, full_partition, lot_id_map = self.extract_lots(
            bin_inv, quadra_cnt, gray,
            line_sensitivity=line_sensitivity,
            sort_mode=sort_mode,
            clean_bldgs_mask=clean_bldgs_mask
        )

        # Calibração fina automática por cota conhecida do lote padrão se informada
        if ref_cota_meters and float(ref_cota_meters) > 0:
            ref_cota = float(ref_cota_meters)
            widths = []
            for cnt in lot_cnts:
                x, y, w, h = cv2.boundingRect(cnt)
                if h > 1.8 * w and w > 20:
                    widths.append(w)
            if widths:
                median_w_px = float(np.median(widths))
                self.meters_per_pixel = ref_cota / median_w_px
                self.scale_denom = round(self.meters_per_pixel / (0.0254 / self.dpi), 1)

        quadra_poly = self.to_metric_polygon(quadra_cnt, origin_bbox, buffer_snap=False)

        # 3. Extrair a geometria de linhas limpa do traçado dos lotes (Zero Gap, Zero duplicatas)
        gdf_divisas, divisa_features_pixel = self.extract_divisas(
            full_partition=full_partition,
            quadra_cnt=quadra_cnt,
            origin_bbox=origin_bbox,
            quadra_code=quadra_code,
            bairro=bairro,
            h_img=h_img
        )

        # GeoJSON de Pixel para overlay Leaflet (apenas linhas do traçado)
        pixel_features = list(divisa_features_pixel)

        # Montar GeoDataFrames
        gdf_quadra = gpd.GeoDataFrame([{
            "ID_QUADRA": quadra_code,
            "BAIRRO": bairro,
            "AREA_M2": round(quadra_poly.area, 2),
            "PERIM_M": round(quadra_poly.length, 2),
            "TOTAL_LINHAS": len(gdf_divisas),
            "ESCALA": f"1:{int(self.scale_denom)}",
            "geometry": quadra_poly
        }], crs=f"EPSG:{self.epsg}")

        gdf_lots = gpd.GeoDataFrame(columns=["NUM_LOTE", "geometry"], crs=f"EPSG:{self.epsg}")

        # Gerar imagem de preview destacando o traçado das divisas em vermelho sobre a planta original
        vis = img.copy()
        cv2.drawContours(vis, [quadra_cnt], -1, (255, 120, 0), 2)

        # Traçar linhas vermelhas de divisas no preview (sem badges ou números de lote)
        for _, row in gdf_divisas.iterrows():
            cv_pts = row.get("cv_pts")
            if cv_pts and len(cv_pts) >= 2:
                pts_arr = np.array(cv_pts, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(vis, [pts_arr], False, (0, 0, 255), 2)

        extensao_divisas_m = round(float(gdf_divisas["COMPR_M"].sum()), 2) if not gdf_divisas.empty else 0.0

        stats = {
            "quadra_code": quadra_code,
            "bairro": bairro or "Não informado",
            "quadra_area_m2": round(quadra_poly.area, 2),
            "quadra_perim_m": round(quadra_poly.length, 2),
            "total_divisas": len(gdf_divisas),
            "extensao_divisas_m": extensao_divisas_m,
            "scale": f"1:{int(self.scale_denom)}",
            "resolution_m_px": round(self.meters_per_pixel, 4),
            "dimensions_px": [w_img, h_img]
        }

        geojson_pixel = {
            "type": "FeatureCollection",
            "features": pixel_features
        }

        return gdf_quadra, gdf_lots, gdf_divisas, vis, img, stats, geojson_pixel
