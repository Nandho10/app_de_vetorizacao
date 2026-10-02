import os
import sys
import uuid
import json
import shutil
import cv2
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory

# Adicionar pasta app ao path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, CURRENT_DIR)

from core.processor import QuadraProcessor
from core.exporter import VectorExporter

app = Flask(__name__, template_folder=os.path.join(CURRENT_DIR, "templates"),
            static_folder=os.path.join(CURRENT_DIR, "static"))

# Diretórios de trabalho
UPLOAD_DIR = os.path.join(CURRENT_DIR, "storage", "uploads")
CACHE_DIR = os.path.join(CURRENT_DIR, "storage", "cache")
OUTPUT_DIR = os.path.join(CURRENT_DIR, "storage", "outputs")
EXAMPLES_DIR = os.path.join(PROJECT_DIR, "Exemplos de quadras")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cache de tarefas em memória
TASKS = {}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/examples", methods=["GET"])
def get_examples():
    """Lista arquivos de exemplo disponíveis na pasta de exemplos."""
    if not os.path.exists(EXAMPLES_DIR):
        return jsonify([])
    files = []
    for f in os.listdir(EXAMPLES_DIR):
        ext = os.path.splitext(f)[1].lower()
        if ext in [".jpg", ".png", ".tif", ".pdf", ".jpeg"] and not f.endswith("_preview.jpg"):
            files.append({
                "filename": f,
                "size_kb": round(os.path.getsize(os.path.join(EXAMPLES_DIR, f)) / 1024, 1)
            })
    return jsonify(files)


@app.route("/api/load_example", methods=["POST"])
def load_example():
    """Carrega um arquivo da pasta de exemplos para processamento."""
    data = request.json or {}
    filename = data.get("filename")
    if not filename:
        return jsonify({"error": "Nome do arquivo não fornecido."}), 400

    src_path = os.path.join(EXAMPLES_DIR, filename)
    if not os.path.exists(src_path):
        return jsonify({"error": "Arquivo não encontrado."}), 404

    task_id = str(uuid.uuid4())[:8]
    task_upload_dir = os.path.join(UPLOAD_DIR, task_id)
    os.makedirs(task_upload_dir, exist_ok=True)

    dest_path = os.path.join(task_upload_dir, filename)
    shutil.copy2(src_path, dest_path)

    TASKS[task_id] = {
        "file_path": dest_path,
        "filename": filename,
        "status": "uploaded"
    }

    return jsonify({"success": True, "task_id": task_id, "filename": filename})


@app.route("/api/upload", methods=["POST"])
def upload():
    """Recebe upload do usuário (.pdf, .tif, .jpg, .png)."""
    if "file" not in request.files:
        return jsonify({"error": "Nenhum arquivo enviado."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Arquivo vazio."}), 400

    task_id = str(uuid.uuid4())[:8]
    task_upload_dir = os.path.join(UPLOAD_DIR, task_id)
    os.makedirs(task_upload_dir, exist_ok=True)

    file_path = os.path.join(task_upload_dir, file.filename)
    file.save(file_path)

    TASKS[task_id] = {
        "file_path": file_path,
        "filename": file.filename,
        "status": "uploaded"
    }

    return jsonify({"success": True, "task_id": task_id, "filename": file.filename})


@app.route("/api/process", methods=["POST"])
def process():
    """Executa a vetorização com os parâmetros escolhidos."""
    data = request.json or {}
    task_id = data.get("task_id")
    if not task_id or task_id not in TASKS:
        return jsonify({"error": "Sessão inválida ou expirada."}), 400

    file_path = TASKS[task_id]["file_path"]
    scale_denom = float(data.get("scale", 750))
    dpi = float(data.get("dpi", 300))
    quadra_code = data.get("quadra_code", "QD-01").strip() or "QD-01"
    bairro = data.get("bairro", "").strip()
    line_sens = float(data.get("sensitivity", 45))
    min_lot_area = float(data.get("min_lot_area", 10.0))
    epsg = int(data.get("epsg", 31983))
    sort_mode = data.get("sort_mode", "cadastral_ring")
    ref_cota = data.get("ref_cota")
    if ref_cota is not None and str(ref_cota).strip() != "":
        ref_cota = float(ref_cota)
    else:
        ref_cota = None

    try:
        processor = QuadraProcessor(
            scale_denom=scale_denom,
            dpi=dpi,
            min_lot_area_m2=min_lot_area,
            epsg=epsg
        )

        gdf_quadra, gdf_lots, vis_img, orig_img, stats, geojson_pixel = processor.process(
            file_path=file_path,
            quadra_code=quadra_code,
            bairro=bairro,
            scale_override=scale_denom,
            line_sensitivity=line_sens,
            sort_mode=sort_mode,
            ref_cota_meters=ref_cota
        )

        # Salvar imagens de cache para o visualizador web
        task_cache_dir = os.path.join(CACHE_DIR, task_id)
        os.makedirs(task_cache_dir, exist_ok=True)

        orig_path = os.path.join(task_cache_dir, "original.jpg")
        prev_path = os.path.join(task_cache_dir, "preview.jpg")

        cv2.imwrite(orig_path, orig_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        cv2.imwrite(prev_path, vis_img, [cv2.IMWRITE_JPEG_QUALITY, 90])

        # Exportar todos os formatos vetoriais
        task_out_dir = os.path.join(OUTPUT_DIR, task_id)
        exported_files = VectorExporter.export_all(
            gdf_quadra=gdf_quadra,
            gdf_lots=gdf_lots,
            output_dir=task_out_dir,
            base_name=f"{quadra_code}_vetorizado",
            crs_epsg=epsg
        )

        # Preparar dados tabulares dos lotes
        lots_table = []
        if not gdf_lots.empty:
            for _, row in gdf_lots.iterrows():
                lots_table.append({
                    "num_lote": row.get("NUM_LOTE", ""),
                    "area_m2": row.get("AREA_M2", 0),
                    "perim_m": row.get("PERIM_M", 0)
                })

        TASKS[task_id].update({
            "status": "processed",
            "exported_files": exported_files,
            "stats": stats
        })

        return jsonify({
            "success": True,
            "task_id": task_id,
            "stats": stats,
            "geojson_pixel": geojson_pixel,
            "lots": lots_table,
            "orig_url": f"/cache/{task_id}/original.jpg",
            "prev_url": f"/cache/{task_id}/preview.jpg"
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Erro durante a vetorização: {str(e)}"}), 500


@app.route("/cache/<task_id>/<filename>")
def serve_cache(task_id, filename):
    """Serve imagens geradas para a interface web."""
    dir_path = os.path.join(CACHE_DIR, task_id)
    return send_from_directory(dir_path, filename)


@app.route("/api/download/<fmt>/<task_id>")
def download(fmt, task_id):
    """Endpoint de download de arquivos vetoriais."""
    if task_id not in TASKS or "exported_files" not in TASKS[task_id]:
        return jsonify({"error": "Arquivo não encontrado para este processamento."}), 404

    files = TASKS[task_id]["exported_files"]
    if fmt not in files:
        return jsonify({"error": f"Formato '{fmt}' não disponível."}), 400

    target_path = files[fmt]
    filename = os.path.basename(target_path)

    mimetypes = {
        "shapefile": "application/zip",
        "gpkg": "application/octet-stream",
        "geojson": "application/geo+json",
        "kml": "application/vnd.google-earth.kml+xml"
    }

    return send_file(
        target_path,
        as_attachment=True,
        download_name=filename,
        mimetype=mimetypes.get(fmt, "application/octet-stream")
    )


if __name__ == "__main__":
    port = 5055
    print(f"\n=======================================================")
    print(f"🚀 Servidor do Vetorizador de Quadras iniciado!")
    print(f"👉 Acesse no navegador: http://localhost:{port}")
    print(f"=======================================================\n")
    app.run(host="0.0.0.0", port=port, debug=False)
