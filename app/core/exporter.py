import os
import shutil
import zipfile
import tempfile
import geopandas as gpd
from osgeo import ogr, osr


class VectorExporter:
    """
    Exportador multiformato para dados vetoriais de quadras, lotes e edificações.
    Formatos suportados: Shapefile (.zip), GeoPackage (.gpkg), GeoJSON (.geojson), KML (.kml).
    """

    @staticmethod
    def export_all(gdf_quadra, gdf_lots, output_dir, base_name="quadra_vetorizada", crs_epsg=None, gdf_edificacoes=None, gdf_divisas=None):
        """
        Gera todos os formatos e retorna um dicionário com os caminhos dos arquivos gerados.
        """
        os.makedirs(output_dir, exist_ok=True)
        results = {}

        has_edif = gdf_edificacoes is not None and not gdf_edificacoes.empty
        has_divisas = gdf_divisas is not None and not gdf_divisas.empty

        # 1. GeoPackage (.gpkg)
        gpkg_path = os.path.join(output_dir, f"{base_name}.gpkg")
        if os.path.exists(gpkg_path):
            os.remove(gpkg_path)
        gdf_quadra.to_file(gpkg_path, layer="quadra", driver="GPKG")
        if not gdf_lots.empty:
            gdf_lots.to_file(gpkg_path, layer="lotes", driver="GPKG")
        if has_divisas:
            gdf_divisas.to_file(gpkg_path, layer="divisas", driver="GPKG")
        if has_edif:
            gdf_edificacoes.to_file(gpkg_path, layer="edificacoes", driver="GPKG")
        results["gpkg"] = gpkg_path

        # 2. GeoJSON (.geojson)
        geojson_path = os.path.join(output_dir, f"{base_name}.geojson")
        features = []
        for _, row in gdf_quadra.iterrows():
            d = row.to_dict()
            d["TIPO_CAMADA"] = "QUADRA"
            features.append(d)
        if not gdf_lots.empty:
            for _, row in gdf_lots.iterrows():
                d = row.to_dict()
                d["TIPO_CAMADA"] = "LOTE"
                features.append(d)
        if has_divisas:
            for _, row in gdf_divisas.iterrows():
                d = row.to_dict()
                d["TIPO_CAMADA"] = "DIVISA"
                features.append(d)
        if has_edif:
            for _, row in gdf_edificacoes.iterrows():
                d = row.to_dict()
                d["TIPO_CAMADA"] = "EDIFICACAO"
                features.append(d)
        gdf_combined = gpd.GeoDataFrame(features, crs=gdf_quadra.crs)
        gdf_combined.to_file(geojson_path, driver="GeoJSON")
        results["geojson"] = geojson_path

        # 3. Shapefile (.zip)
        shp_zip_path = os.path.join(output_dir, f"{base_name}_shapefile.zip")
        with tempfile.TemporaryDirectory() as temp_shp_dir:
            shp_q = os.path.join(temp_shp_dir, "quadra.shp")
            gdf_quadra.to_file(shp_q)
            if not gdf_lots.empty:
                shp_l = os.path.join(temp_shp_dir, "lotes.shp")
                gdf_lots.to_file(shp_l)
            if has_divisas:
                shp_d = os.path.join(temp_shp_dir, "divisas.shp")
                gdf_divisas.to_file(shp_d)
            if has_edif:
                shp_e = os.path.join(temp_shp_dir, "edificacoes.shp")
                gdf_edificacoes.to_file(shp_e)
            
            with zipfile.ZipFile(shp_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in os.listdir(temp_shp_dir):
                    zf.write(os.path.join(temp_shp_dir, f), arcname=f)
        results["shapefile"] = shp_zip_path

        # 4. KML (.kml)
        kml_path = os.path.join(output_dir, f"{base_name}.kml")
        try:
            with open(kml_path, "w", encoding="utf-8") as f:
                f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
                f.write('<kml xmlns="http://www.opengis.net/kml/2.2">\n')
                f.write('<Document>\n')
                f.write(f'  <name>{base_name}</name>\n')
                
                # Styles
                f.write('  <Style id="quadraStyle"><LineStyle><color>ff0000ff</color><width>3</width></LineStyle><PolyStyle><fill>0</fill></PolyStyle></Style>\n')
                f.write('  <Style id="loteStyle"><LineStyle><color>ff00aa00</color><width>1.5</width></LineStyle><PolyStyle><color>4000ff00</color></PolyStyle></Style>\n')
                f.write('  <Style id="divisaStyle"><LineStyle><color>ff0000ff</color><width>2.5</width></LineStyle></Style>\n')
                f.write('  <Style id="edifExistenteStyle"><LineStyle><color>ff00bfff</color><width>2</width></LineStyle><PolyStyle><color>8000ffff</color></PolyStyle></Style>\n')
                f.write('  <Style id="edifDemolidaStyle"><LineStyle><color>ff0000ff</color><width>2</width></LineStyle><PolyStyle><color>600000ff</color></PolyStyle></Style>\n')

                # Base origin in local metric
                ref_lat = -23.4862
                ref_lon = -46.3485
                m_to_lat = 1.0 / 111320.0
                m_to_lon = 1.0 / (111320.0 * 0.917)

                # Quadra Placemark
                for _, row in gdf_quadra.iterrows():
                    poly = row.geometry
                    f.write('  <Placemark>\n')
                    f.write(f'    <name>Quadra {row.get("ID_QUADRA", "")}</name>\n')
                    f.write(f'    <description>Área: {row.get("AREA_M2", "")} m²</description>\n')
                    f.write('    <styleUrl>#quadraStyle</styleUrl>\n')
                    f.write('    <Polygon><outerBoundaryIs><LinearRing><coordinates>\n')
                    if poly.geom_type == 'Polygon':
                        for x, y in poly.exterior.coords:
                            lon = ref_lon + x * m_to_lon
                            lat = ref_lat + y * m_to_lat
                            f.write(f'      {lon:.7f},{lat:.7f},0\n')
                    f.write('    </coordinates></LinearRing></outerBoundaryIs></Polygon>\n')
                    f.write('  </Placemark>\n')

                # Divisas (Linhas Vermelhas) Placemarks
                if has_divisas:
                    for _, row in gdf_divisas.iterrows():
                        line = row.geometry
                        if line.geom_type == 'LineString':
                            id_div = row.get("ID_DIVISA", "")
                            tipo = row.get("TIPO", "")
                            lote_a = row.get("LOTE_A", "")
                            lote_b = row.get("LOTE_B", "")
                            compr = row.get("COMPR_M", "")
                            f.write('  <Placemark>\n')
                            f.write(f'    <name>{id_div} ({tipo})</name>\n')
                            f.write(f'    <description>Tipo: {tipo} | Entre: Lote {lote_a} e Lote {lote_b} | Extensão: {compr} m</description>\n')
                            f.write('    <styleUrl>#divisaStyle</styleUrl>\n')
                            f.write('    <LineString><coordinates>\n')
                            for x, y in line.coords:
                                lon = ref_lon + x * m_to_lon
                                lat = ref_lat + y * m_to_lat
                                f.write(f'      {lon:.7f},{lat:.7f},0\n')
                            f.write('    </coordinates></LineString>\n')
                            f.write('  </Placemark>\n')

                # Lotes Placemarks
                if not gdf_lots.empty:
                    for _, row in gdf_lots.iterrows():
                        poly = row.geometry
                        f.write('  <Placemark>\n')
                        f.write(f'    <name>Lote {row.get("NUM_LOTE", "")}</name>\n')
                        f.write(f'    <description>Área: {row.get("AREA_M2", "")} m² | Perímetro: {row.get("PERIM_M", "")} m</description>\n')
                        f.write('    <styleUrl>#loteStyle</styleUrl>\n')
                        f.write('    <Polygon><outerBoundaryIs><LinearRing><coordinates>\n')
                        if poly.geom_type == 'Polygon':
                            for x, y in poly.exterior.coords:
                                lon = ref_lon + x * m_to_lon
                                lat = ref_lat + y * m_to_lat
                                f.write(f'      {lon:.7f},{lat:.7f},0\n')
                        f.write('    </coordinates></LinearRing></outerBoundaryIs></Polygon>\n')
                        f.write('  </Placemark>\n')

                # Edificações Placemarks
                if has_edif:
                    for _, row in gdf_edificacoes.iterrows():
                        poly = row.geometry
                        tipo = row.get("TIPO", "EXISTENTE")
                        style_id = "#edifExistenteStyle" if tipo == "EXISTENTE" else "#edifDemolidaStyle"
                        status_str = "Área Construída" if tipo == "EXISTENTE" else "Área Livre (Demolida)"
                        f.write('  <Placemark>\n')
                        f.write(f'    <name>{row.get("ID_EDIF", "")} ({tipo})</name>\n')
                        f.write(f'    <description>Tipo: {tipo} ({status_str}) | Lote: {row.get("LOTE", "")} | Área: {row.get("AREA_M2", "")} m²</description>\n')
                        f.write(f'    <styleUrl>{style_id}</styleUrl>\n')
                        f.write('    <Polygon><outerBoundaryIs><LinearRing><coordinates>\n')
                        if poly.geom_type == 'Polygon':
                            for x, y in poly.exterior.coords:
                                lon = ref_lon + x * m_to_lon
                                lat = ref_lat + y * m_to_lat
                                f.write(f'      {lon:.7f},{lat:.7f},0\n')
                        f.write('    </coordinates></LinearRing></outerBoundaryIs></Polygon>\n')
                        f.write('  </Placemark>\n')

                f.write('</Document>\n')
                f.write('</kml>\n')
            results["kml"] = kml_path
        except Exception as e:
            print("Erro ao gerar KML:", e)

        return results
