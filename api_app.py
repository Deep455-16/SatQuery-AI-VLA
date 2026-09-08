import os
import tempfile
import uuid
import shutil
import logging
import time
from io import BytesIO
import base64
from flask import Flask, request, jsonify, g
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("satquery.api")

from satquery.controller.agent_controller import AgentController, ExecutionResult
from satquery.utils.image_io import load_image

app = Flask(__name__, static_folder='frontend/dist', static_url_path='/')
CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '*')
CORS(app, resources={r"/api/*": {"origins": CORS_ORIGINS}})

MAX_UPLOAD_SIZE_MB = int(os.environ.get('MAX_UPLOAD_SIZE_MB', 50))
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_SIZE_MB * 1024 * 1024
MAX_IMAGES = int(os.environ.get('MAX_IMAGES_PER_REQUEST', 10))

ALLOWED_MIMETYPES = {'image/tiff', 'image/png', 'image/jpeg'}

def error_response(code: str, message: str, status_code: int = 400):
    return jsonify({'error_code': code, 'error': message}), status_code

@app.before_request
def before_request():
    g.request_id = str(uuid.uuid4())[:8]
    g.start_time = time.time()

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def pil_to_b64(img):
    if img is None:
        return None
    try:
        buf = BytesIO()
        img.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None

def safe_meta(meta) -> dict:
    if meta is None:
        return {}
    return {
        "filename":         getattr(meta, "path", "").replace("\\", "/").split("/")[-1],
        "modality":         getattr(meta, "modality", "unknown"),
        "width":            getattr(meta, "width", None),
        "height":           getattr(meta, "height", None),
        "bands":            getattr(meta, "bands", None),
        "crs":              getattr(meta, "crs", None),
        "resolution":       (f"{meta.resolution_x:.1f} m" if getattr(meta, "resolution_x", None) else "—"),
        "acquisition_date": getattr(meta, "acquisition_date", None),
        "sensor":           getattr(meta, "sensor", None),
        "format":           getattr(meta, "format", ""),
        "is_sentinel2":     bool(getattr(meta, "is_sentinel2", False)),
        "is_sar":           bool(getattr(meta, "is_sar", False)),
    }

# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "version": "SatQuery AI 1.0.0", "timestamp": time.time()})

@app.route("/api/models/status", methods=["GET"])
def get_models_status():
    from satquery.adapters.gemini_adapter import get_adapter
    gemini_status = {"status": "UNAVAILABLE"}
    try:
        if get_adapter().is_available():
            gemini_status = {"status": "CONFIGURED", "model": "gemini-2.5-flash"}
    except Exception:
        pass

    terraq_status = {"status": "STANDBY", "lora": False, "note": "Standby ready — set VLM_MODEL_PATH to activate"}
    try:
        from satquery.model.terraq_vl import get_terraq_vl
        tq = get_terraq_vl()
        terraq_status["status"] = tq.status if tq.status != "UNAVAILABLE" else "STANDBY"
        terraq_status["lora"] = bool(getattr(tq, 'lora_path', ''))
        terraq_status["note"] = tq.get_status_dict().get("note", "Standby ready")
    except Exception:
        pass

    import psutil
    hw = {
        "cpu": f"{psutil.cpu_percent()}%",
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 1),
        "cuda": False,
        "gpu": None
    }
    try:
        import torch
        if torch.cuda.is_available():
            hw["cuda"] = True
            hw["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass

    return jsonify({
        "terraq_vl": terraq_status,
        "gemini": gemini_status,
        "hardware": hw
    })

@app.route("/api/status", methods=["GET"])
def get_status():
    from satquery.adapters.gemini_adapter import get_adapter
    gemini_ok = False
    try:
        gemini_ok = get_adapter().is_available()
    except Exception:
        pass

    terraq_ok = False
    try:
        from satquery.model.terraq_vl import get_terraq_vl
        terraq_ok = get_terraq_vl().is_available()
    except Exception:
        pass

    return jsonify({
        "ai_engine": "operational" if gemini_ok else "unavailable",
        "terraq_vl": "standby",
        "gemini": "active" if gemini_ok else "unavailable",
        "api": "connected",
        "version": "SatQuery AI 1.0.0",
    })

@app.route("/api/analyze/multiple", methods=["POST"])
def analyze_multiple():
    return do_analyze("Multiple Images")

@app.route("/api/analyze", methods=["POST"])
def analyze():
    mode = request.form.get("mode", "Single Image")
    return do_analyze(mode)

def do_analyze(mode):
    if "files" not in request.files:
        return error_response("NO_FILES", "No files uploaded.", 400)

    uploaded = request.files.getlist("files")
    if not uploaded or all(f.filename == "" for f in uploaded):
        return error_response("NO_FILES", "File list is empty.", 400)

    if len(uploaded) > MAX_IMAGES:
        return error_response("TOO_MANY_FILES", f"Max images exceeded: {MAX_IMAGES}", 400)

    query      = request.form.get("query", "").strip()
    session_id = request.form.get("session_id") or str(uuid.uuid4())

    if not query:
        return error_response("EMPTY_QUERY", "Query is required.", 400)

    tmp_dir = tempfile.mkdtemp(prefix="satquery_")
    images_data = []

    try:
        for upload in uploaded:
            if upload.mimetype not in ALLOWED_MIMETYPES:
                return error_response("UNSUPPORTED_FORMAT", f"Unsupported mime type: {upload.mimetype}", 415)

            safe_name = "".join([c for c in upload.filename if c.isalpha() or c.isdigit() or c in ' .-_']).rstrip()
            if not safe_name:
                safe_name = "upload.img"
            dst = os.path.join(tmp_dir, safe_name)
            upload.save(dst)

            try:
                arr, meta = load_image(dst)
                images_data.append((arr, meta))
            except Exception:
                return error_response("INVALID_IMAGE", f"Could not read '{safe_name}'", 422)

        if not images_data:
            return error_response("NO_VALID_IMAGES", "No valid imagery could be loaded.", 422)

        logger.info(
            "Analysis started: query=%r mode=%s files=%d req_id=%s",
            query, mode, len(images_data), g.request_id
        )
        controller = AgentController()

        try:
            result: ExecutionResult = controller.handle_query(query, images_data, session_id)
        except Exception as e:
            logger.exception("Agent controller failed during handle_query")
            return error_response("ANALYSIS_FAILED", f"Agent controller failed: {str(e)}", 500)

        res = result.to_dict() if hasattr(result, 'to_dict') else {"answer": getattr(result, "answer", "No answer")}

        res["rgb_visualization"]         = pil_to_b64(getattr(result, "rgb_visualization", None))
        res["false_color_visualization"] = pil_to_b64(getattr(result, "false_color_visualization", None))
        res["change_map"]                = pil_to_b64(getattr(result, "change_map_pil", None))
        res["detection_visualization"]   = pil_to_b64(getattr(result, "detection_visualization", None))

        res["images_metadata"] = [safe_meta(m) for _, m in images_data]
        res["execution_trace"] = getattr(result, "execution_trace", {})
        res["confidence"]      = getattr(result, "confidence", None)
        res["image_ids"]       = []
        res["analysis_id"]     = None

        # Save to JSON history (without MongoDB)
        try:
            from satquery.database.mongodb import save_analysis
            filenames = [m.get("filename") for m in res.get("images_metadata", [])]
            save_analysis(session_id, query, res, images=filenames)
        except Exception as e:
            logger.warning(f"Could not save history: {e}")

        logger.info(
            "Analysis complete: task=%s latency=%dms req_id=%s model=%s",
            getattr(result, 'task', 'unknown'),
            getattr(result, 'processing_time_ms', 0),
            g.request_id,
            ", ".join(getattr(result, 'models_used', []) or ["unknown"]),
        )
        return jsonify(res)

    except Exception:
        logger.exception("Unhandled error in do_analyze")
        return error_response("INTERNAL_ERROR", "Internal Server Error", 500)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

@app.route("/api/history", methods=["GET"])
def get_history():
    from satquery.database.mongodb import get_analysis_history
    limit = int(request.args.get('limit', 50))
    raw = get_analysis_history(limit)
    flattened = []
    for item in raw:
        res = item.get("result", {})
        flattened.append({
            "id":               item.get("id") or item.get("_id"),
            "session_id":       item.get("session_id"),
            "timestamp":        item.get("timestamp"),
            "query":            item.get("query", ""),
            "images":           item.get("images", []),
            # flatten result fields
            "task":             res.get("task", "vqa"),
            "answer":           res.get("answer", ""),
            "observations":     res.get("observations", []),
            "inferences":       res.get("inferences", []),
            "limitations":      res.get("limitations", []),
            "evidence_strength":res.get("evidence_strength", ""),
            "confidence":       res.get("confidence"),
            "models_used":      res.get("models_used", []),
            "processing_time_ms": res.get("processing_time_ms", 0),
            "images_metadata":  res.get("images_metadata", []),
            "warnings":         res.get("warnings", []),
        })
    return jsonify({"history": flattened})

# ──────────────────────────────────────────────
# VLM on-demand loader
# ──────────────────────────────────────────────
@app.route("/api/vlm/load", methods=["POST"])
def load_vlm_model():
    """Trigger on-demand loading of TerraQ-VL model."""
    try:
        from satquery.model.terraq_vl import get_terraq_vl
        tq = get_terraq_vl()
        result = tq.load_model_now()
        return jsonify({'status': tq.status, 'note': tq.get_status_dict().get('note', '')})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ──────────────────────────────────────────────
# Serve React Frontend
# ──────────────────────────────────────────────
@app.route('/', defaults={'path': ''})
@app.route('/analysis')
@app.route('/status')
@app.route('/history')
@app.route('/about')
@app.route('/<path:path>')
def serve_react(path=""):
    if app.static_folder and os.path.exists(app.static_folder):
        if path != "" and not path.startswith("api/") and os.path.exists(os.path.join(app.static_folder, path)):
            return app.send_static_file(path)
        else:
            return app.send_static_file('index.html')
    else:
        return jsonify({'error': 'Frontend build not found. Run "npm run build" in the frontend directory.'}), 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
