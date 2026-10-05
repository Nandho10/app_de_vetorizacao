import os
import sys
sys.path.append(r"D:\Documentos\Luiz\Antigravity\Vetorizador de quadras")
from app.core.processor import QuadraProcessor
from app.core.exporter import VectorExporter

test_files = [
    (r"D:\Documentos\Luiz\Antigravity\Vetorizador de quadras\Exemplos de quadras\44454-42-39.jpg", 750, 300, "44454"),
    (r"D:\Documentos\Luiz\Antigravity\Vetorizador de quadras\Exemplos de quadras\44463-33-38.png", 1000, 300, "44463"),
]

out_dir = r"D:\Documentos\Luiz\Antigravity\Vetorizador de quadras\output\verification_test"
os.makedirs(out_dir, exist_ok=True)

for path, scale, dpi, name in test_files:
    print(f"=== Testing {name} ===")
    qp = QuadraProcessor(scale_denom=scale, dpi=dpi)
    gdf_q, gdf_l, gdf_e, vis, img, stats, geo_px = qp.process(path)
    
    print(f"Quadra bounds: {gdf_q.geometry.iloc[0].bounds}")
    print(f"Total lots: {len(gdf_l)}")
    print(f"Total edificacoes: {len(gdf_e) if gdf_e is not None else 0}")
    if gdf_e is not None and not gdf_e.empty:
        existentes = len(gdf_e[gdf_e['TIPO'] == 'EXISTENTE'])
        demolidas = len(gdf_e[gdf_e['TIPO'] == 'DEMOLIDA'])
        print(f"  -> Existentes: {existentes}, Demolidas: {demolidas}")
    print(f"Stats: {stats}")
    
    # Test export
    exp_files = VectorExporter.export_all(gdf_q, gdf_l, out_dir, base_name=name, gdf_edificacoes=gdf_e)
    print("Exported files:")
    for k, v in exp_files.items():
        size = os.path.getsize(v) if os.path.exists(v) else 0
        print(f"  {k}: {os.path.basename(v)} ({size} bytes)")
    print()

print("ALL VERIFICATIONS COMPLETED SUCCESSFULLY!")
