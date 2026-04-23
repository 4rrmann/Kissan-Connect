"""
predictor.py — Kissan Connect inference
Supports both:
  - CNN model  (disease_cnn_model.keras + disease_class_names.pkl)  ← preferred
  - HOG model  (disease_model.pkl + disease_label_encoder.pkl)       ← fallback

Bug fixes (2025):
  1. Yield: Season/State/Crop inputs are now normalized (.strip().title())
     before encoding, matching training preprocessing.
  2. Yield: Crop_Year was missing from the feature vector (model expects 5
     features; old code sent 4). Fixed by adding crop_year argument.
  3. Disease: _preprocess_for_cnn used a bare "except: return None", hiding
     the real error. Now returns (array, error_str) and surfaces exceptions.
  4. Disease: predict_image called preprocess_input twice — once explicitly
     and once inside the model (it is baked in as a Keras layer). The model
     received all-near-(-1) inputs for every image. Fixed by removing the
     explicit call; the model layer handles normalization internally.
  5. Disease: Image.open(filepath) is now done via io.BytesIO after reading
     the file bytes first. This avoids the TensorFlow/Pillow JPEG decoder
     conflict on Windows (TF hijacks libjpeg when imported before PIL).
"""

import os
import warnings
from datetime import datetime

import joblib
import numpy as np

warnings.filterwarnings("ignore", message=".*MT19937.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*BitGenerator.*", category=UserWarning)

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(_BASE_DIR, "model")

IMG_SIZE = 224    # MobileNetV2 input size

# Default Crop_Year used when the caller doesn't supply one.
# Set to a year that was present during training (dataset ends at 2015)
# so the model interpolates rather than extrapolates.
_DEFAULT_CROP_YEAR = 2013


def _mpath(*p):
    return os.path.join(MODEL_DIR, *p)


# ══════════════════════════════════════════════════════════════
#  Input normalisation helpers
# ══════════════════════════════════════════════════════════════

def _normalise(value: str) -> str:
    """
    Strip whitespace and title-case a string.
    Matches the exact preprocessing applied during training:
        df["Season"] = df["Season"].str.strip().str.title()
        df["State_Name"] = df["State_Name"].str.strip().str.title()
    """
    return value.strip().title()


def _safe_encode(le, raw_value: str, field_name: str):
    """
    Normalise → encode.  Returns (encoded_int, None) on success or
    (None, error_str) with allowed values listed on failure.

    Accepted input variations (all resolve to the same encoded value):
        "kharif"   →  normalise  →  "Kharif"  →  encode  →  3
        "KHARIF"   →  normalise  →  "Kharif"  →  encode  →  3
        "kHaRiF"   →  normalise  →  "Kharif"  →  encode  →  3
        "Kharif  " →  normalise  →  "Kharif"  →  encode  →  3
    """
    normalised = _normalise(raw_value)
    known = list(le.classes_)

    print(f"[predictor] {field_name}: raw={repr(raw_value)}  normalised={repr(normalised)}")
    print(f"[predictor] Known {field_name} values: {known}")

    if normalised not in le.classes_:
        return None, (
            f"Unknown {field_name} '{raw_value}'. "
            f"Allowed values: {known}"
        )

    encoded = int(le.transform([normalised])[0])
    return encoded, None


# ══════════════════════════════════════════════════════════════
#  Disease prediction  (CNN preferred, HOG fallback)
# ══════════════════════════════════════════════════════════════
_cnn_model        = None
_cnn_classes      = None
_hog_clf          = None
_hog_le           = None
_disease_error    = None
_disease_mode     = None   # "cnn" | "hog" | None


def get_disease_model_error():
    return _disease_error


def reset_disease_model():
    """Force reload on next call (use after retraining)."""
    global _cnn_model, _cnn_classes, _hog_clf, _hog_le, _disease_error, _disease_mode
    _cnn_model = _cnn_classes = _hog_clf = _hog_le = _disease_error = _disease_mode = None


def _try_load_cnn():
    """Try to load CNN model. Returns True on success."""
    global _cnn_model, _cnn_classes, _disease_error, _disease_mode

    model_path  = _mpath("disease_cnn_model.keras")
    names_path  = _mpath("disease_class_names.pkl")

    if not os.path.isfile(model_path) or not os.path.isfile(names_path):
        return False   # not trained yet — fall through to HOG

    try:
        from tensorflow import keras
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _cnn_model   = keras.models.load_model(model_path)
            _cnn_classes = joblib.load(names_path)
        _disease_mode = "cnn"
        print(f"[predictor] CNN model loaded — {len(_cnn_classes)} classes")
        return True
    except Exception as exc:
        _disease_error = f"CNN load failed: {exc}"
        return False


def _try_load_hog():
    """Try to load HOG/sklearn model. Returns True on success."""
    global _hog_clf, _hog_le, _disease_error, _disease_mode

    clf_path = _mpath("disease_model.pkl")
    le_path  = _mpath("disease_label_encoder.pkl")

    if not os.path.isfile(clf_path) or not os.path.isfile(le_path):
        _disease_error = (
            "No disease model found.\n"
            "Train the CNN model first:\n"
            "  pip install tensorflow\n"
            "  python train_disease_model.py --dataset datasets/plantvillage"
        )
        return False

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _hog_clf = joblib.load(clf_path)
            _hog_le  = joblib.load(le_path)
        _disease_mode = "hog"
        print(f"[predictor] HOG model loaded — {len(_hog_le.classes_)} classes")
        return True
    except Exception as exc:
        err = str(exc)
        if "MT19937" in err or "BitGenerator" in err:
            _disease_error = (
                "Disease model is incompatible with your NumPy version.\n"
                "Retrain it: python train_disease_model.py --dataset datasets/plantvillage"
            )
        else:
            _disease_error = f"Failed to load disease model: {exc}"
        return False


def _ensure_disease_model():
    """Load CNN first, fall back to HOG, cache result."""
    global _disease_mode
    if _disease_mode is not None:
        return _disease_mode
    if _try_load_cnn():
        return "cnn"
    if _try_load_hog():
        return "hog"
    return None


# ── CNN feature helpers ───────────────────────────────────────
def _preprocess_for_cnn(filepath: str):
    """
    Load and resize an image for MobileNetV2 inference.
    Returns (array, None) on success or (None, error_str) on failure.

    WHY BytesIO:
      TensorFlow bundles its own libjpeg. On Windows, importing TF before
      PIL can cause TF to hijack PIL's JPEG decoder, making Image.open()
      silently fail or crash. Reading the file into a bytes buffer and
      passing that to PIL via io.BytesIO bypasses the decoder conflict.

    WHY no preprocess_input here:
      train_disease_model.py bakes preprocess_input INSIDE the Keras model:
          x = keras.applications.mobilenet_v2.preprocess_input(inputs)
      Calling it again would apply it twice:
          [0,255] -> [-1,+1] (1st, inside model) -> all near -1 (2nd, here)
      Just pass raw float32 pixels (0-255); the model layer handles the rest.
    """
    import io

    if not os.path.isfile(filepath):
        return None, f"Uploaded file not found: {filepath}"
    try:
        from PIL import Image

        # Read bytes first to bypass TF/PIL JPEG decoder conflict on Windows
        with open(filepath, "rb") as f:
            img_bytes = f.read()

        img = (Image.open(io.BytesIO(img_bytes))
                    .convert("RGB")
                    .resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS))
        arr = np.array(img, dtype=np.float32)   # (224, 224, 3) range [0,255]
        result = np.expand_dims(arr, 0)          # (1, 224, 224, 3)
        print(f"[predictor] Image loaded: shape={result.shape}  "
              f"range=[{arr.min():.0f},{arr.max():.0f}]")
        return result, None

    except ImportError:
        return None, "Pillow not installed. Run: pip install pillow"
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


# ── HOG feature helpers ───────────────────────────────────────
_HOG_SIZE = 128

def _preprocess_for_hog(filepath: str) -> "np.ndarray | None":
    try:
        import cv2
        from PIL import Image
        pil = Image.open(filepath).convert("RGB").resize((_HOG_SIZE, _HOG_SIZE))
        arr = np.array(pil, dtype=np.uint8)
        img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hog_desc = cv2.HOGDescriptor(
            (_HOG_SIZE, _HOG_SIZE), (16,16), (8,8), (8,8), 9)
        hog_f    = hog_desc.compute(gray).flatten()

        hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_h  = cv2.calcHist([hsv],[0],None,[32],[0,180]).flatten()
        s_h  = cv2.calcHist([hsv],[1],None,[32],[0,256]).flatten()
        v_h  = cv2.calcHist([hsv],[2],None,[32],[0,256]).flatten()
        blur = cv2.GaussianBlur(gray,(5,5),0)
        edg  = cv2.Canny(blur,50,150)
        e_h  = cv2.calcHist([edg],[0],None,[16],[0,256]).flatten()

        feat = np.concatenate([hog_f,h_h,s_h,v_h,e_h]).astype(np.float32)
        return feat.reshape(1, -1)
    except Exception:
        return None


# ── Format class name for display ────────────────────────────
def _fmt(raw: str) -> str:
    """
    'Tomato___Late_blight'         →  'Tomato Late Blight'
    'Corn_(maize)___Common_rust_'  →  'Corn Common Rust'
    """
    name = raw.replace("___", " ").replace("_", " ").replace("(", "").replace(")", "")
    return " ".join(name.split()).title()


# ── Public predict function ───────────────────────────────────
def predict_image(filepath: str):
    """
    Returns (display_name: str, confidence: float) or (None, None).
    Never raises — all exceptions caught internally.
    """
    global _disease_error

    mode = _ensure_disease_model()
    if mode is None:
        return None, None

    # ── CNN path ─────────────────────────────────────────────
    if mode == "cnn":
        # _preprocess_for_cnn now returns (array, error) — never swallows exceptions
        arr, img_err = _preprocess_for_cnn(filepath)
        if arr is None:
            # Surface the REAL error (e.g. FileNotFoundError, JPEG decode fail)
            _disease_error = img_err or "Could not open or process the image."
            print(f"[predictor] Image preprocessing failed: {_disease_error}")
            return None, None
        try:
            # DO NOT call preprocess_input here.
            # The model already contains it as a Keras layer (baked in during training):
            #   x = keras.applications.mobilenet_v2.preprocess_input(inputs)
            # Calling it again would compress every image to all-near-(-1), breaking inference.
            preds = _cnn_model.predict(arr, verbose=0)[0]
            idx   = int(np.argmax(preds))
            conf  = float(preds[idx] * 100)
            label = _fmt(_cnn_classes[idx])
            print(f"[predictor] Prediction: {label}  confidence={conf:.1f}%")
            return label, round(conf, 1)
        except Exception as exc:
            _disease_error = f"CNN inference error: {exc}"
            print(f"[predictor] CNN inference failed: {exc}")
            return None, None

    # ── HOG path ─────────────────────────────────────────────
    if mode == "hog":
        feat = _preprocess_for_hog(filepath)
        if feat is None:
            _disease_error = "Could not open or process the image."
            return None, None
        try:
            if hasattr(_hog_clf, "predict_proba"):
                proba = _hog_clf.predict_proba(feat)[0]
                idx   = int(np.argmax(proba))
                conf  = float(proba[idx] * 100)
            else:
                idx  = int(_hog_clf.predict(feat)[0])
                conf = 60.0
            label = _fmt(_hog_le.inverse_transform([idx])[0])
            return label, round(conf, 1)
        except Exception as exc:
            _disease_error = f"HOG inference error: {exc}"
            return None, None

    return None, None


# ══════════════════════════════════════════════════════════════
#  Crop recommendation
# ══════════════════════════════════════════════════════════════
_crop_model      = None
_crop_le         = None
_crop_load_error = None


def predict_crop_recommendation(features):
    """
    features: list of 7 floats [N, P, K, temperature, humidity, ph, rainfall]
    Returns (crop_name: str, None) on success or (None, error_str) on failure.
    """
    global _crop_model, _crop_le, _crop_load_error

    if _crop_load_error:
        return None, _crop_load_error

    if _crop_model is None:
        mp = _mpath("crop_recommendation_model.pkl")
        lp = _mpath("label_encoder.pkl")
        if not os.path.isfile(mp) or not os.path.isfile(lp):
            _crop_load_error = "Crop model missing. Run: python train_models.py"
            return None, _crop_load_error
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _crop_model = joblib.load(mp)
                _crop_le    = joblib.load(lp)
        except Exception as exc:
            _crop_load_error = str(exc)
            return None, _crop_load_error

    try:
        pred = _crop_model.predict(np.array([features]))
        return _crop_le.inverse_transform(pred)[0], None
    except Exception as exc:
        return None, str(exc)


# ══════════════════════════════════════════════════════════════
#  Yield prediction
# ══════════════════════════════════════════════════════════════
_yield_model      = None
_yield_encoders   = None
_yield_load_error = None


def get_yield_model_error():
    return _yield_load_error


def predict_crop_yield(
    state: str,
    crop: str,
    season: str,
    area: float,
    crop_year: int = _DEFAULT_CROP_YEAR,
):
    """
    Predict crop production in tonnes.

    Parameters
    ----------
    state      : State name — normalised automatically (strip + title-case).
                 e.g. "punjab", "PUNJAB", "Punjab " all work.
    crop       : Crop name — normalised automatically.
                 e.g. "wheat", "Wheat", " WHEAT " all work.
    season     : Season — normalised automatically.
                 e.g. "kharif", "KHARIF", "kHaRiF", "Kharif " all work.
                 Allowed values (post-normalisation):
                   Autumn | Kharif | Rabi | Summer | Whole Year | Winter
    area       : Cultivated area in hectares (must be > 0).
    crop_year  : Year of cultivation. Defaults to 2013 (mid-range of
                 training data 1997–2015).  Pass the actual year when
                 available for slightly better accuracy.

    Returns
    -------
    float          — predicted production in tonnes (≥ 0.0) on success
    str            — descriptive error message on validation failure
    None           — model files missing or failed to load
    """
    global _yield_model, _yield_encoders, _yield_load_error

    # ── Load models (once, cached) ────────────────────────────
    if _yield_load_error:
        return None

    if _yield_model is None:
        mp = _mpath("crop_yield_model.pkl")
        ep = _mpath("label_encoders.pkl")
        if not os.path.isfile(mp) or not os.path.isfile(ep):
            _yield_load_error = "Yield model missing. Run: python train_models.py"
            return None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _yield_model    = joblib.load(mp)
                _yield_encoders = joblib.load(ep)
            print("[predictor] Yield model loaded successfully.")
        except Exception as exc:
            _yield_load_error = str(exc)
            return None

    le_state, le_crop, le_season = _yield_encoders

    # ── Input validation ──────────────────────────────────────
    if area <= 0:
        return "Area must be greater than 0 hectares."

    # ── Normalise + encode  (strip + title-case, matching training) ───────
    #
    # Training pipeline:
    #   df["State_Name"] = df["State_Name"].str.strip().str.title()
    #   df["Crop"]       = df["Crop"].str.strip()          # casing preserved
    #   df["Season"]     = df["Season"].str.strip().str.title()
    #
    # NOTE: Crop preserves original mixed casing from the dataset, so we
    # only strip (no title-case) to stay consistent with training.

    state_norm  = state.strip().title()
    crop_norm   = crop.strip()           # title() NOT applied — matches training
    season_norm = season.strip().title()

    print(f"[predictor] Normalised inputs → "
          f"State: {repr(state_norm)}  "
          f"Crop: {repr(crop_norm)}  "
          f"Season: {repr(season_norm)}  "
          f"Area: {area}  "
          f"Crop_Year: {crop_year}")
    print(f"[predictor] Known seasons: {list(le_season.classes_)}")
    print(f"[predictor] Known states : {list(le_state.classes_)}")

    # Encode state
    se, err = _safe_encode(le_state, state_norm, "State")
    if err:
        return err

    # Encode crop
    ce, err = _safe_encode(le_crop, crop_norm, "Crop")
    if err:
        return err

    # Encode season
    sne, err = _safe_encode(le_season, season_norm, "Season")
    if err:
        return err

    # ── Feature vector — must match training order exactly ────
    #   Training: [state_enc, crop_enc, season_enc, Area, Crop_Year]
    features = np.array([[se, ce, sne, float(area), float(crop_year)]])
    print(f"[predictor] Feature vector: {features}")

    try:
        result = float(_yield_model.predict(features)[0])
        result = max(0.0, round(result, 2))
        print(f"[predictor] Raw prediction: {result:.2f} tonnes")
        return result
    except Exception as exc:
        return f"Prediction error: {exc}"
