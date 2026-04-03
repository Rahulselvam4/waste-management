# ─────────────────────────────────────────────────────────────────
# model_builder.py
# Rebuilds the MobileNetV2 architecture locally and loads weights.
# This avoids ALL Keras version mismatch errors when loading .keras
# files saved in a different TF version (e.g. Colab TF2.19 vs local).
# ─────────────────────────────────────────────────────────────────

import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Dropout, GlobalAveragePooling2D, BatchNormalization, Input
)
from tensorflow.keras.optimizers import Adam

CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
NUM_CLASSES  = len(CLASS_NAMES)
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "waste_weights_FINAL.weights.h5")
KERAS_PATH   = os.path.join(os.path.dirname(__file__), "waste_model_FINAL.keras")

_model = None


def build_architecture():
    """
    Rebuilds the exact same architecture used in training.
    Must match Part 2 training code exactly.
    """
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = True

    inputs = Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def load_model_safe():
    """
    Tries multiple loading strategies in order of compatibility.
    Strategy 1: Load weights into rebuilt architecture (most reliable)
    Strategy 2: Direct load with custom_objects (fallback)
    Strategy 3: TF SavedModel format (last resort)
    """
    global _model
    if _model is not None:
        return _model

    # ── Strategy 1: Weights file (recommended) ────────────────────
    if os.path.exists(WEIGHTS_PATH):
        print("Loading via weights file (strategy 1)...")
        model = build_architecture()
        model.load_weights(WEIGHTS_PATH)
        _model = model
        print("Model loaded successfully via weights!")
        return _model

    # ── Strategy 2: Direct .keras load ────────────────────────────
    if os.path.exists(KERAS_PATH):
        print("Trying direct .keras load (strategy 2)...")
        try:
            _model = tf.keras.models.load_model(KERAS_PATH)
            print("Model loaded successfully via direct load!")
            return _model
        except Exception as e:
            print(f"Direct load failed: {e}")
            print("Trying with safe_mode=False...")
            try:
                _model = tf.keras.models.load_model(KERAS_PATH, safe_mode=False)
                print("Model loaded with safe_mode=False!")
                return _model
            except Exception as e2:
                print(f"Safe_mode=False also failed: {e2}")

    # ── Strategy 3: SavedModel ─────────────────────────────────────
    savedmodel_path = os.path.join(os.path.dirname(__file__), "waste_savedmodel")
    if os.path.exists(savedmodel_path):
        print("Trying SavedModel format (strategy 3)...")
        _model = tf.saved_model.load(savedmodel_path)
        print("SavedModel loaded!")
        return _model

    raise FileNotFoundError(
        "\n\n"
        "═══════════════════════════════════════════════\n"
        "  MODEL FILE NOT FOUND\n"
        "═══════════════════════════════════════════════\n"
        "Please do ONE of the following:\n\n"
        "Option A (Recommended):\n"
        "  1. Open your Colab Part 2 notebook\n"
        "  2. Run the weight export cell\n"
        "  3. Download waste_weights_FINAL.weights.h5\n"
        "  4. Place it in this project folder\n\n"
        "Option B:\n"
        "  Place waste_model_FINAL.keras in this folder\n"
        "═══════════════════════════════════════════════\n"
    )
