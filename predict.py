# ─────────────────────────────────────────────────────────────────
# predict.py — Standalone inference module
# Usage: python predict.py --image path/to/image.jpg
# ─────────────────────────────────────────────────────────────────

import numpy as np
import cv2
import os
import argparse
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from model_builder import load_model_safe

CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
IMG_SIZE    = (224, 224)


def preprocess_image(image_input):
    """
    Accepts: file path (str), uploaded file object, PIL Image, numpy array.
    Returns: numpy array shape (1, 224, 224, 3) ready for model.predict()
    """
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            raise ValueError(f"Cannot read image: {image_input}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    elif hasattr(image_input, 'read'):
        # Streamlit UploadedFile or file-like object
        file_bytes = np.asarray(bytearray(image_input.read()), dtype=np.uint8)
        img        = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img        = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_input.seek(0)

    else:
        img = np.array(image_input)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[-1] == 4:
            img = img[:, :, :3]

    img = cv2.resize(img, IMG_SIZE).astype('float32')
    img = preprocess_input(img)              # scales [0,255] → [-1,+1]
    return np.expand_dims(img, axis=0)       # (1,224,224,3)


def predict(image_input):
    """
    Classifies a single waste image.

    Returns dict:
        predicted_class  (str)   — e.g. 'plastic'
        confidence       (float) — 0.0 to 100.0
        all_probabilities(dict)  — {class: probability}
        top3             (list)  — [(class, prob), ...]
    """
    model     = load_model_safe()
    processed = preprocess_image(image_input)
    raw_probs = model.predict(processed, verbose=0)[0]

    all_probs       = {n: float(p) for n, p in zip(CLASS_NAMES, raw_probs)}
    predicted_idx   = int(np.argmax(raw_probs))
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence      = float(raw_probs[predicted_idx]) * 100
    top3            = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)[:3]

    return {
        "predicted_class":    predicted_class,
        "confidence":         round(confidence, 2),
        "all_probabilities":  all_probs,
        "top3":               top3,
    }


def predict_batch(files):
    """Classifies multiple images efficiently."""
    model  = load_model_safe()
    batch, valid_flags = [], []

    for f in files:
        try:
            arr = preprocess_image(f)
            batch.append(arr[0])
            valid_flags.append(True)
        except Exception as e:
            print(f"Skipping: {e}")
            batch.append(np.zeros((224, 224, 3), dtype='float32'))
            valid_flags.append(False)

    raw_preds = model.predict(np.stack(batch), verbose=0)
    results   = []

    for probs, valid in zip(raw_preds, valid_flags):
        if not valid:
            results.append({"error": "Could not process image"})
            continue
        all_probs       = {n: float(p) for n, p in zip(CLASS_NAMES, probs)}
        predicted_idx   = int(np.argmax(probs))
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence      = float(probs[predicted_idx]) * 100
        top3            = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)[:3]
        results.append({
            "predicted_class":   predicted_class,
            "confidence":        round(confidence, 2),
            "all_probabilities": all_probs,
            "top3":              top3,
        })

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WasteAI — Inference CLI")
    parser.add_argument("--image", required=True)
    args   = parser.parse_args()

    result = predict(args.image)
    print(f"\n{'═'*40}")
    print(f"  Prediction : {result['predicted_class'].upper()}")
    print(f"  Confidence : {result['confidence']:.1f}%")
    print(f"{'─'*40}")
    for cls, prob in sorted(result['all_probabilities'].items(),
                            key=lambda x: x[1], reverse=True):
        bar = "█" * int(prob * 25)
        print(f"  {cls:12s}: {prob*100:5.1f}%  {bar}")
    print(f"{'═'*40}\n")
