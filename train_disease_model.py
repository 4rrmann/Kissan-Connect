"""
train_disease_model.py  -Kissan Connect CNN Disease Trainer
============================================================
Trains a CNN on a PlantVillage-style dataset (any number of classes).
Uses Transfer Learning: MobileNetV2 fine-tuned on your data.

SETUP:
  pip install tensorflow pillow numpy

USAGE  -recommended (point at the dataset ROOT):
  python train_disease_model.py --data_dir datasets/plantvillage

USAGE  -point directly at a split folder if auto-detect fails:
  python train_disease_model.py --data_dir "datasets/plantvillage/New Plant Diseases Dataset(Augmented)/valid"

The script auto-detects your folder layout:

  Layout A  -separate train/ and valid/ splits (ideal):
    <root>/train/<class_folders>/
    <root>/valid/<class_folders>/

  Layout B  -only one labelled split (your current situation):
    <root>/valid/<class_folders>/      ← used as training source
    An 80 / 20 train-validation split is created automatically.

  Layout C  -split inside a sub-folder (e.g. augmented dataset zip):
    <root>/New Plant Diseases Dataset(Augmented)/
        train/<class_folders>/
        valid/<class_folders>/

  All layouts are handled without any manual path editing.

Output:
  model/disease_cnn_model.keras
  model/disease_class_names.pkl

Bugs fixed vs original version:
  1. positional arg + required=True → TypeError on startup
     Fixed: changed to --data_dir named argument
  2. args.dataset AttributeError (attr was named 'datasets')
     Fixed: consistent --data_dir / args.data_dir naming
  3. find_train_valid returned (None, None) when no train/ existed
     Fixed: new discover_splits() handles valid-only layouts
  4. valid/ class folders never reached because #3 exited first
     Fixed: single-split layout triggers automatic 80/20 split
  5. test/ at a different depth than valid/ was ignored
     Fixed: depth-agnostic recursive scan up to 4 levels
"""

import os
import sys
import argparse
import warnings
import numpy as np
import joblib
from pathlib import Path

warnings.filterwarnings("ignore", message=".*MT19937.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ── Constants ────────────────────────────────────────────────
IMG_SIZE    = 224      # MobileNetV2 native size
BATCH_SIZE  = 32
EPOCHS_HEAD = 10       # phase 1: train new head only
EPOCHS_FT   = 10       # phase 2: fine-tune last 30 base layers
TRAIN_SPLIT = 0.8      # used only when no separate train/ folder exists

BASE_DIR  = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

# Names that indicate a data split folder (not a class folder)
_SPLIT_NAMES = {"train", "valid", "val", "validation", "test"}


# ════════════════════════════════════════════════════════════════
#  Dataset discovery  (fixes bugs 3, 4, 5)
# ════════════════════════════════════════════════════════════════

def _looks_like_class_dir(d: Path) -> bool:
    """
    A class folder contains image files (or is empty  -could be sparse).
    It does NOT contain sub-folders named like splits (train/valid/test).
    """
    if not d.is_dir():
        return False
    children = list(d.iterdir())
    has_files  = any(f.is_file() for f in children)
    has_splits = any(
        f.is_dir() and f.name.lower() in _SPLIT_NAMES for f in children
    )
    return (has_files or not children) and not has_splits


def _is_split_root(d: Path) -> bool:
    """
    A split root is a folder whose direct children all look like class dirs.
    e.g.  valid/
            Apple___Apple_scab/   ← class dir
            Tomato___Late_blight/ ← class dir
    """
    children = [c for c in d.iterdir() if c.is_dir() and not c.name.startswith(".")]
    return bool(children) and all(_looks_like_class_dir(c) for c in children)


def _discover_splits(root: Path, depth: int = 0) -> dict[str, Path]:
    """
    Recursively walk `root` (up to 4 levels deep).
    Returns a dict mapping normalised split name → Path.
      e.g. {"train": Path(...), "valid": Path(...)}

    Handles wrapper folders like "New Plant Diseases Dataset(Augmented)/"
    transparently  -they are traversed but not treated as splits.
    """
    if depth > 4:
        return {}

    found: dict[str, Path] = {}
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue

        name_lower = d.name.lower()

        if _is_split_root(d):
            # Use the normalised name (val → valid) as the key
            key = "valid" if name_lower in {"val", "validation"} else name_lower
            if key in _SPLIT_NAMES:
                class_count = sum(1 for c in d.iterdir() if c.is_dir() and not c.name.startswith("."))
                print(f"    ✔ Split detected : '{d.name}'  →  {class_count} classes  [{d}]")
                found[key] = d
            # Don't recurse into confirmed split roots  -their children are classes
        else:
            # Could be a wrapper folder  -recurse
            deeper = _discover_splits(d, depth + 1)
            found.update(deeper)

    return found


def resolve_dataset(data_dir: Path):
    """
    Given the user-supplied root, return (train_dir, valid_dir, use_split).

    use_split is True when train_dir == valid_dir and Keras must create the
    validation set via validation_split= rather than validation_data=.

    Strategy
    --------
    1. If data_dir itself is a split root (user pointed directly at valid/):
       treat it as the only available split.
    2. Otherwise scan recursively for train/ and valid/ sub-folders.
    3. If only one split is found, use it for both train and val with an
       80 / 20 Keras validation_split (no data leakage: Keras takes the
       last 20% of each class alphabetically, not randomly).
    """
    print(f"\n  Scanning dataset root: {data_dir}")

    # Case: user pointed directly at a split folder (e.g. .../valid)
    if _is_split_root(data_dir):
        key = data_dir.name.lower()
        key = "valid" if key in {"val", "validation"} else key
        class_count = sum(1 for c in data_dir.iterdir() if c.is_dir())
        print(f"    ✔ Using supplied path as split root  ({class_count} classes)")
        splits = {key: data_dir}
    else:
        splits = _discover_splits(data_dir)

    if not splits:
        _die_no_splits(data_dir)

    # Best-case: both train and valid found
    if "train" in splits and "valid" in splits:
        print(f"\n  Layout: separate train / valid splits  ✅")
        return splits["train"], splits["valid"], False

    # Acceptable: only one split found → use with 80/20 Keras split
    # Preference order: train > valid > test
    for key in ("train", "valid", "test"):
        if key in splits:
            src = splits[key]
            print(f"\n  Layout: single split ('{src.name}' only)")
            print(f"  ⚠  No separate 'train/' folder found.")
            print(f"     Will train on '{src.name}/' with an automatic")
            print(f"     {int(TRAIN_SPLIT*100)}/{int((1-TRAIN_SPLIT)*100)} train-validation split.")
            return src, src, True

    _die_no_splits(data_dir)


def _die_no_splits(data_dir: Path) -> None:
    print(f"\n❌  Could not find any usable split inside: {data_dir}")
    print("    Expected one of these layouts:")
    print()
    print("    Layout A  -separate splits:")
    print("      <root>/train/<Apple___Apple_scab>/  ...")
    print("      <root>/valid/<Apple___Apple_scab>/  ...")
    print()
    print("    Layout B  -single split (valid-only):")
    print("      <root>/valid/<Apple___Apple_scab>/  ...")
    print()
    print("    Layout C  -inside a sub-folder:")
    print("      <root>/New Plant Diseases Dataset(Augmented)/valid/<classes>/")
    print()
    print("    Tip: run with --data_dir pointing at the folder that")
    print("    CONTAINS the split folders (train/, valid/, etc.).")
    sys.exit(1)


# ════════════════════════════════════════════════════════════════
#  Class discovery
# ════════════════════════════════════════════════════════════════

def get_class_names(split_dir: Path) -> list[str]:
    """
    Read class names from folder names inside split_dir.
    Validates that at least one class was found.
    """
    classes = sorted(
        d.name for d in split_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )
    if not classes:
        print(f"\n❌  No class sub-folders found inside: {split_dir}")
        print("    Each sub-folder should be a disease/plant class, e.g.:")
        print("      Apple___Apple_scab/")
        print("      Tomato___Late_blight/")
        sys.exit(1)

    print(f"\n  Discovered {len(classes)} classes:")
    for c in classes:
        print(f"    ✔  {c}")
    return classes


# ════════════════════════════════════════════════════════════════
#  Keras dataset pipelines
# ════════════════════════════════════════════════════════════════

def make_datasets(
    train_dir: Path,
    valid_dir: Path,
    class_names: list[str],
    use_split: bool,
):
    """
    Returns (train_ds, valid_ds).

    use_split=True  → single source dir; Keras creates the val split via
                       validation_split=0.2 (last 20% per class, deterministic).
    use_split=False → separate train/ and valid/ dirs.
    """
    import tensorflow as tf

    AUTOTUNE = tf.data.AUTOTUNE

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomContrast(0.1),
        tf.keras.layers.RandomBrightness(0.1),
    ], name="augmentation")

    common_kwargs = dict(
        class_names=class_names,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        seed=42,
    )

    if use_split:
        # Both train and valid come from the same folder  -Keras splits it.
        train_ds = tf.keras.utils.image_dataset_from_directory(
            str(train_dir),
            shuffle=True,
            validation_split=1.0 - TRAIN_SPLIT,
            subset="training",
            **common_kwargs,
        )
        valid_ds = tf.keras.utils.image_dataset_from_directory(
            str(valid_dir),
            shuffle=False,
            validation_split=1.0 - TRAIN_SPLIT,
            subset="validation",
            **common_kwargs,
        )
    else:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            str(train_dir),
            shuffle=True,
            **common_kwargs,
        )
        valid_ds = tf.keras.utils.image_dataset_from_directory(
            str(valid_dir),
            shuffle=False,
            **common_kwargs,
        )

    train_ds = (
        train_ds
        .map(lambda x, y: (data_augmentation(x, training=True), y),
             num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
    )
    valid_ds = valid_ds.prefetch(AUTOTUNE)
    return train_ds, valid_ds


# ════════════════════════════════════════════════════════════════
#  Model
# ════════════════════════════════════════════════════════════════

def build_model(num_classes: int):
    """MobileNetV2 with a custom classification head."""
    import tensorflow as tf
    from tensorflow import keras

    base = keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False   # frozen for phase 1

    inputs  = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x       = keras.applications.mobilenet_v2.preprocess_input(inputs)
    x       = base(x, training=False)
    x       = keras.layers.GlobalAveragePooling2D()(x)
    x       = keras.layers.Dropout(0.3)(x)
    x       = keras.layers.Dense(256, activation="relu")(x)
    x       = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs), base


# ════════════════════════════════════════════════════════════════
#  Training
# ════════════════════════════════════════════════════════════════

def train(
    train_dir: Path,
    valid_dir: Path,
    class_names: list[str],
    use_split: bool,
):
    from tensorflow import keras

    print(f"\n  Building MobileNetV2 for {len(class_names)} classes ...")
    model, base = build_model(len(class_names))

    train_ds, valid_ds = make_datasets(train_dir, valid_dir, class_names, use_split)

    callbacks = [
        keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2, verbose=1),
        keras.callbacks.ModelCheckpoint(
            str(MODEL_DIR / "disease_cnn_best.keras"),
            save_best_only=True,
            monitor="val_accuracy",
            verbose=1,
        ),
    ]

    # ── Phase 1: head only ──────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Phase 1  -Training head  ({EPOCHS_HEAD} epochs max, base frozen)")
    print(f"{'='*55}")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(train_ds, validation_data=valid_ds,
              epochs=EPOCHS_HEAD, callbacks=callbacks)

    # ── Phase 2: fine-tune last 30 base layers ──────────────
    print(f"\n{'='*55}")
    print(f"  Phase 2  -Fine-tuning top layers  ({EPOCHS_FT} epochs max)")
    print(f"{'='*55}")
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(train_ds, validation_data=valid_ds,
              epochs=EPOCHS_FT, callbacks=callbacks)

    print("\n  Evaluating on validation set ...")
    loss, acc = model.evaluate(valid_ds, verbose=0)
    print(f"  ✅  Final validation accuracy: {acc * 100:.1f}%")
    return model


# ════════════════════════════════════════════════════════════════
#  Save
# ════════════════════════════════════════════════════════════════

def save(model, class_names: list[str]) -> None:
    model_path = MODEL_DIR / "disease_cnn_model.keras"
    names_path = MODEL_DIR / "disease_class_names.pkl"

    model.save(str(model_path))
    joblib.dump(class_names, str(names_path))

    print(f"\n  ✅  Model saved   → {model_path}")
    print(f"  ✅  Classes saved → {names_path}  ({len(class_names)} classes)")
    print(f"\n  Restart your Flask app  -the disease detector is ready.")


# ════════════════════════════════════════════════════════════════
#  CLI  (fixes bugs 1 & 2)
# ════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        prog="train_disease_model.py",
        description="Train Kissan Connect CNN plant-disease classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Point at the dataset root (recommended  -auto-detects layout):
  python train_disease_model.py --data_dir datasets/plantvillage

  # Point directly at the split folder when auto-detect is not needed:
  python train_disease_model.py --data_dir "datasets/plantvillage/New Plant Diseases Dataset(Augmented)/valid"

Dataset layout support
----------------------
  A) Separate splits:    <root>/train/<classes>/   <root>/valid/<classes>/
  B) Valid-only:         <root>/valid/<classes>/   (auto 80/20 split)
  C) Inside sub-folder:  <root>/SubFolder/train/   <root>/SubFolder/valid/
        """,
    )

    # FIX 1: named argument (--data_dir) instead of positional
    # FIX 2: attribute name matches consistently (args.data_dir everywhere)
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        metavar="PATH",
        help=(
            "Path to the dataset root (or directly to a split folder). "
            "Example: datasets/plantvillage"
        ),
    )
    return parser.parse_args()


# ════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()   # FIX 1 & 2: no TypeError, correct attribute

    # ── Validate path exists ────────────────────────────────
    data_dir = Path(args.data_dir)   # FIX 2: args.data_dir not args.dataset
    if not data_dir.exists():
        print(f"\n❌  Path not found: {data_dir}")
        print(f"    Check the --data_dir value and try again.")
        sys.exit(1)
    if not data_dir.is_dir():
        print(f"\n❌  Path is not a directory: {data_dir}")
        sys.exit(1)

    # ── TensorFlow check ────────────────────────────────────
    try:
        import tensorflow as tf
        print(f"  TensorFlow {tf.__version__} detected  ✅")
        gpus = tf.config.list_physical_devices("GPU")
        print(f"  GPUs available: {len(gpus)}")
    except ImportError:
        print("❌  TensorFlow not installed.")
        print("    Run: pip install tensorflow")
        sys.exit(1)

    print("=" * 55)
    print("  KISSAN CONNECT  -CNN Disease Model Trainer")
    print("=" * 55)

    # ── Discover dataset layout (fixes 3, 4, 5) ─────────────
    train_dir, valid_dir, use_split = resolve_dataset(data_dir)

    print(f"\n  Train source : {train_dir}")
    print(f"  Valid source : {valid_dir}")
    if use_split:
        print(f"  Split mode   : validation_split={1.0 - TRAIN_SPLIT:.0%} (auto)")

    # ── Discover classes ────────────────────────────────────
    class_names = get_class_names(train_dir)

    # ── Train & save ────────────────────────────────────────
    model = train(train_dir, valid_dir, class_names, use_split)
    save(model, class_names)


if __name__ == "__main__":
    main()
