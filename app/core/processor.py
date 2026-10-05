import os
import cv2
import numpy as np
import tempfile
import zipfile
from PIL import Image
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.ops import polygonize, unary_union
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
        return approx

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
            return []

        # Tesselacao Voronoi via Distance Transform euclidiano (Zero Gap)
        _, (row_idx, col_idx) = ndi.distance_transform_edt(filtered_labels == 0, return_indices=True)
        full_partition = filtered_labels[row_idx, col_idx]
        full_partition[quadra_mask == 0] = 0

        valid_lots = []
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        for lid in range(1, current_id):
            lot_mask = (full_partition == lid).astype(np.uint8) * 255
            lot_mask = cv2.morphologyEx(lot_mask, cv2.MORPH_CLOSE, kernel_close)
            lot_mask = ndi.binary_fill_holes(lot_mask).astype(np.uint8) * 255
            
            cnts, _ = cv2.findContours(lot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                cnt = max(cnts, key=cv2.contourArea)
                arc = cv2.arcLength(cnt, True)
                epsilon = max(1.0, 0.0035 * arc)
                approx_cnt = cv2.approxPolyDP(cnt, epsilon, True)
                if len(approx_cnt) >= 3:
                    valid_lots.append(approx_cnt)

        # Ordenação espacial: Anel Cadastral (Padrão de Loteamento) ou Varredura Linear
        if sort_mode == "cadastral_ring":
            bx, by, bw, bh = cv2.boundingRect(quadra_cnt)
            center_y = by + bh / 2.0
            top = []
            right = []
            bottom = []
            for c in valid_lots:
                M = cv2.moments(c)
                cx = M["m10"] / M["m00"] if M["m00"] > 0 else 0
                cy = M["m01"] / M["m00"] if M["m00"] > 0 else 0
                if cx > bx + 0.82 * bw:
                    if cy < by + 0.35 * bh:
                        top.append((cx, cy, c))
                    else:
                        right.append((cx, cy, c))
                elif cy < center_y:
                    top.append((cx, cy, c))
                else:
                    bottom.append((cx, cy, c))
            top.sort(key=lambda x: x[0])     # Topo: esquerda -> direita (1..N)
            right.sort(key=lambda x: x[1])   # Lateral: topo -> base
            bottom.sort(key=lambda x: -x[0]) # Base: direita -> esquerda
            valid_lots = [item[2] for item in top + right + bottom]
        else:
            def get_centroid(c):
                M = cv2.moments(c)
                if M["m00"] > 0:
                    return (round(M["m01"] / M["m00"], -1), M["m10"] / M["m00"])
                return (0, 0)
            valid_lots.sort(key=get_centroid)

        return valid_lots

    def to_metric_polygon(self, contour, origin_bbox):
        """Converte coordenadas de pixels de imagem para coordenadas cartesianas em metros reais."""
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
        lot_cnts = self.extract_lots(
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

        quadra_poly = self.to_metric_polygon(quadra_cnt, origin_bbox)

        lot_records = []
        pixel_features = []

        # Adicionar Quadra no GeoJSON de Pixel para overlay
        pixel_features.append({
            "type": "Feature",
            "properties": {
                "TIPO": "QUADRA",
                "ID": quadra_code,
                "AREA_M2": round(quadra_poly.area, 2),
                "PERIM_M": round(quadra_poly.length, 2),
                "COLOR": "#0284c7"
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [self.to_pixel_polygon(quadra_cnt, h_img)]
            }
        })

        lot_polys = []
        for i, cnt in enumerate(lot_cnts, start=1):
            p = self.to_metric_polygon(cnt, origin_bbox)
            if p.area >= self.min_lot_area_m2:
                num_lote = f"{i:02d}"
                lot_polys.append((num_lote, p, cnt))
                lot_records.append({
                    "NUM_LOTE": num_lote,
                    "QUADRA": quadra_code,
                    "BAIRRO": bairro,
                    "AREA_M2": round(p.area, 2),
                    "PERIM_M": round(p.length, 2),
                    "geometry": p
                })
                pixel_features.append({
                    "type": "Feature",
                    "properties": {
                        "TIPO": "LOTE",
                        "ID": num_lote,
                        "AREA_M2": round(p.area, 2),
                        "PERIM_M": round(p.length, 2),
                        "COLOR": "#16a34a"
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [self.to_pixel_polygon(cnt, h_img)]
                    }
                })

        # Montar GeoDataFrames
        gdf_quadra = gpd.GeoDataFrame([{
            "ID_QUADRA": quadra_code,
            "BAIRRO": bairro,
            "AREA_M2": round(quadra_poly.area, 2),
            "PERIM_M": round(quadra_poly.length, 2),
            "QTD_LOTES": len(lot_records),
            "ESCALA": f"1:{int(self.scale_denom)}",
            "geometry": quadra_poly
        }], crs=f"EPSG:{self.epsg}")

        if lot_records:
            gdf_lots = gpd.GeoDataFrame(lot_records, crs=f"EPSG:{self.epsg}")
        else:
            gdf_lots = gpd.GeoDataFrame(columns=["NUM_LOTE", "QUADRA", "BAIRRO", "AREA_M2", "PERIM_M", "geometry"], crs=f"EPSG:{self.epsg}")

        # Gerar imagem de preview
        vis = img.copy()
        cv2.drawContours(vis, [quadra_cnt], -1, (255, 120, 0), 4)
        
        np.random.seed(42)
        overlay = vis.copy()
        for i, cnt in enumerate(lot_cnts, start=1):
            color = [int(c) for c in np.random.randint(60, 220, size=3)]
            cv2.drawContours(overlay, [cnt], -1, color, -1)
            cv2.drawContours(vis, [cnt], -1, (0, 220, 50), 2)
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(vis, str(i), (cx-10, cy+6), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        cv2.addWeighted(overlay, 0.30, vis, 0.70, 0, vis)

        area_media_lote = round(sum(l["AREA_M2"] for l in lot_records) / max(1, len(lot_records)), 2)

        stats = {
            "quadra_code": quadra_code,
            "bairro": bairro or "Não informado",
            "quadra_area_m2": round(quadra_poly.area, 2),
            "quadra_perim_m": round(quadra_poly.length, 2),
            "total_lots": len(lot_records),
            "area_media_lote_m2": area_media_lote,
            "scale": f"1:{int(self.scale_denom)}",
            "resolution_m_px": round(self.meters_per_pixel, 4),
            "dimensions_px": [w_img, h_img]
        }

        geojson_pixel = {
            "type": "FeatureCollection",
            "features": pixel_features
        }

        return gdf_quadra, gdf_lots, vis, img, stats, geojson_pixel
