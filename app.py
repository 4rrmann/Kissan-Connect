"""
app.py — Kissan Connect Flask application
"""
import io
import os
import uuid
import warnings
from datetime import datetime

# Suppress numpy 2.x BitGenerator warning from sklearn model pickles
warnings.filterwarnings(
    "ignore",
    message=".*MT19937.*is not a known BitGenerator.*",
    category=UserWarning,
)

from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename

try:
    from flask_cors import CORS
    _HAS_CORS = True
except ImportError:
    _HAS_CORS = False

from chatbot import generate_response
from predictor import (
    predict_image,
    predict_crop_yield,
    predict_crop_recommendation,
    get_disease_model_error,
    get_yield_model_error,
)

# ─── App setup ────────────────────────────────────────────────
app = Flask(__name__)
if _HAS_CORS:
    CORS(app)

app.secret_key = "kissan-connect-2024"

UPLOAD_FOLDER      = os.path.join(os.path.dirname(__file__), "static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp", "bmp", "tiff"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Valid seasons (must match training preprocessing: .strip().title())
VALID_SEASONS = ["Autumn", "Kharif", "Rabi", "Summer", "Whole Year", "Winter"]

# Crop_Year: training data spans 1997–2015.
# Use a sensible default (mid-point) when the user doesn't supply a year.
_DEFAULT_CROP_YEAR = 2013


def allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


# ─── Routes ───────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/disease-predictor", methods=["GET", "POST"])
def handle_disease_prediction():
    prediction = confidence = image_url = error = None

    if request.method == "POST":
        # ── Validate upload ───────────────────────────────────
        if "image" not in request.files:
            error = "No image uploaded. Please select a file."
            return render_template("crop-disease.html", error=error)

        file = request.files["image"]

        if not file or file.filename == "":
            error = "No file selected. Please choose a crop image."
            return render_template("crop-disease.html", error=error)

        if not allowed_file(file.filename):
            error = "Unsupported file type. Please upload a JPG, PNG, WEBP or GIF image."
            return render_template("crop-disease.html", error=error)

        # ── Save file safely ──────────────────────────────────
        try:
            # secure_filename strips non-ASCII chars (e.g. Hindi filenames)
            # and can return an empty string or just an extension like "jpg".
            # UUID fallback guarantees a valid, unique filename every time.
            raw_name  = secure_filename(file.filename or "")
            # os.path.splitext("jpg") → ("jpg", "") so check both parts
            stem, ext = os.path.splitext(raw_name)
            ext       = ext.lower() if ext else ".jpg"
            # Use original name only when it has BOTH a non-empty stem AND a
            # recognised extension. Fall back to UUID otherwise. This handles:
            #   - Hindi filenames: secure_filename("पत्ता.jpg") → "jpg" (no stem)
            #   - Empty uploads:   filename = ""
            #   - No-extension:    "nodot" → uuid.jpg
            if stem and ext in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff"}:
                filename = raw_name
            else:
                filename = f"{uuid.uuid4().hex}{ext}"
            filepath  = os.path.join(app.config["UPLOAD_FOLDER"], filename)

            # Read into memory first, then write — avoids partial-write issues
            # and ensures the file is fully flushed before PIL tries to open it.
            img_bytes = file.read()
            with open(filepath, "wb") as out:
                out.write(img_bytes)

            # Verify the file was actually written
            if not os.path.isfile(filepath) or os.path.getsize(filepath) == 0:
                raise IOError(f"File saved as 0 bytes at {filepath}")

        except Exception as exc:
            error = f"Could not save the uploaded file: {exc}"
            return render_template("crop-disease.html", error=error)

        # ── Run prediction (never crashes the server) ─────────
        prediction, confidence = predict_image(filepath)
        image_url = url_for("static", filename=f"uploads/{filename}")

        if prediction is None:
            error = get_disease_model_error() or "Disease prediction failed. Please try another image."

    return render_template(
        "crop-disease.html",
        prediction=prediction,
        confidence=confidence,
        image_url=image_url,
        error=error,
    )


@app.route("/crop-predictor", methods=["GET", "POST"])
def handle_crop_recommendation():
    predicted_crop = None

    if request.method == "POST":
        try:
            features = [
                float(request.form["N"]),
                float(request.form["P"]),
                float(request.form["K"]),
                float(request.form["temperature"]),
                float(request.form["humidity"]),
                float(request.form["ph"]),
                float(request.form["rainfall"]),
            ]
            name, err = predict_crop_recommendation(features)
            predicted_crop = name if not err else f"Error: {err}"
        except (KeyError, ValueError) as exc:
            predicted_crop = f"Error: Missing or invalid input — {exc}"
        except Exception as exc:
            predicted_crop = f"Error: {exc}"

    return render_template("form.html", crop=predicted_crop)


@app.route("/yield-predictor", methods=["GET", "POST"])
def handle_yield_prediction():
    prediction = error = None

    if request.method == "POST":
        try:
            # ── Read form inputs ──────────────────────────────
            state  = request.form["state"]
            crop   = request.form["crop"]
            season = request.form["season"]
            area   = float(request.form["area"])

            # ── Crop_Year: use form value if provided and valid,
            #    otherwise fall back to the training-data default.
            raw_year = request.form.get("crop_year", "").strip()
            if raw_year.isdigit():
                crop_year = int(raw_year)
            else:
                crop_year = _DEFAULT_CROP_YEAR

            # ── Call predictor ────────────────────────────────
            # predictor.py normalises state/crop/season internally
            # (.strip().title()), so we pass the raw form values.
            result = predict_crop_yield(state, crop, season, area, crop_year)

            if result is None:
                # Model files missing or failed to load
                error = get_yield_model_error() or "Yield model unavailable. Run train_models.py."
            elif isinstance(result, str):
                # Descriptive validation error from predictor
                # (e.g. "Unknown season. Allowed: [...]")
                error = result
            else:
                # Successful float prediction
                prediction = result

        except (KeyError, ValueError) as exc:
            error = f"Invalid input: {exc}"
        except Exception as exc:
            error = f"Unexpected error: {exc}"

    return render_template(
        "yield.html",
        prediction=prediction,
        error=error,
        valid_seasons=VALID_SEASONS,   # pass to template for dropdown
    )


@app.route("/chat", methods=["POST"])
def handle_query():
    user_message = request.json.get("message", "").strip()
    if not user_message:
        return jsonify({"response": "Please enter a valid query."})
    try:
        bot_response = generate_response(user_message)
    except Exception as exc:
        bot_response = f"Bot error: {exc}"
    return jsonify({"response": bot_response})


# ─── Run ──────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5001)
