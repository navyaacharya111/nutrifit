import os
import numpy as np
import tensorflow as tf
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "food_classifier_food101.keras")
CLASSES_PATH = os.path.join(BASE_DIR, "food101_classes.txt")

# Configurable Unknown / OOD Confidence Threshold
CONFIDENCE_THRESHOLD = 0.40

print("=" * 60)
print("Loading NutriFit AI Model & Classes...")
print("=" * 60)

# Load model at startup
model = None
try:
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print(f"Model loaded successfully from {MODEL_PATH}")
    else:
        raise ValueError("Model file missing")
except Exception as e:
    print(f"Failed to load model on startup ({e}). Generating from weights...")
    try:
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
        from tensorflow.keras.models import Model
        import model_weights

        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        base_model.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        d1 = Dense(256, activation='relu')
        x = d1(x)
        x = Dropout(0.5)(x)
        d2 = Dense(101, activation='softmax')
        predictions = d2(x)

        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Set weights
        d1.set_weights([model_weights.dense1_k, model_weights.dense1_b])
        d2.set_weights([model_weights.dense2_k, model_weights.dense2_b])
        
        # We don't even need to save it to disk, just keep it in memory
        print("Model successfully generated from weights in RAM.")
    except Exception as gen_e:
        print("Failed to generate model from weights:", gen_e)

if model is not None:
    print("Model input shape:", model.input_shape)
    print("Model output shape:", model.output_shape)

FOOD_CLASSES = []
if os.path.exists(CLASSES_PATH):
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        FOOD_CLASSES = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(FOOD_CLASSES)} classes from {CLASSES_PATH}")
else:
    print("WARNING: Classes file not found at", CLASSES_PATH)

if model is not None and len(FOOD_CLASSES) > 0:
    if model.output_shape[-1] != len(FOOD_CLASSES):
        print(f"CRITICAL WARNING: Model output classes ({model.output_shape[-1]}) mismatch with food_classes.txt ({len(FOOD_CLASSES)})")

# ------------------------------------------------------------
# NUTRITION DATABASE (per 100g approximate)
# ------------------------------------------------------------
NUTRITION_DB = {
    "apple_pie": {"calories": 237, "protein": 1.9, "carbs": 34.0, "fat": 11.0},
    "baby_back_ribs": {"calories": 320, "protein": 24.0, "carbs": 0.0, "fat": 25.0},
    "baklava": {"calories": 428, "protein": 6.8, "carbs": 53.0, "fat": 23.0},
    "banana": {"calories": 89, "protein": 1.1, "carbs": 22.8, "fat": 0.3},
    "beef_carpaccio": {"calories": 170, "protein": 20.0, "carbs": 1.0, "fat": 10.0},
    "beef_tartare": {"calories": 210, "protein": 19.0, "carbs": 2.0, "fat": 14.0},
    "beet_salad": {"calories": 120, "protein": 2.5, "carbs": 11.0, "fat": 7.5},
    "beignets": {"calories": 390, "protein": 7.0, "carbs": 48.0, "fat": 19.0},
    "bibimbap": {"calories": 170, "protein": 6.5, "carbs": 26.0, "fat": 4.5},
    "bread_pudding": {"calories": 270, "protein": 5.5, "carbs": 42.0, "fat": 9.5},
    "breakfast_burrito": {"calories": 240, "protein": 10.5, "carbs": 23.0, "fat": 12.0},
    "bruschetta": {"calories": 180, "protein": 4.5, "carbs": 24.0, "fat": 7.5},
    "caesar_salad": {"calories": 190, "protein": 7.0, "carbs": 8.0, "fat": 15.0},
    "cannoli": {"calories": 340, "protein": 5.0, "carbs": 38.0, "fat": 19.0},
    "caprese_salad": {"calories": 175, "protein": 9.0, "carbs": 3.0, "fat": 14.0},
    "carrot_cake": {"calories": 415, "protein": 4.2, "carbs": 54.0, "fat": 21.0},
    "ceviche": {"calories": 110, "protein": 15.0, "carbs": 7.0, "fat": 2.0},
    "cheese_plate": {"calories": 380, "protein": 22.0, "carbs": 4.0, "fat": 30.0},
    "cheesecake": {"calories": 321, "protein": 5.5, "carbs": 25.5, "fat": 22.5},
    "chicken_curry": {"calories": 180, "protein": 12.0, "carbs": 8.0, "fat": 11.0},
    "chicken_quesadilla": {"calories": 290, "protein": 16.0, "carbs": 26.0, "fat": 13.0},
    "chicken_wings": {"calories": 203, "protein": 30.0, "carbs": 0.0, "fat": 8.1},
    "chocolate_cake": {"calories": 371, "protein": 5.3, "carbs": 50.0, "fat": 18.0},
    "chocolate_mousse": {"calories": 310, "protein": 4.5, "carbs": 32.0, "fat": 19.0},
    "churros": {"calories": 450, "protein": 4.5, "carbs": 55.0, "fat": 24.0},
    "clam_chowder": {"calories": 130, "protein": 5.0, "carbs": 12.0, "fat": 7.0},
    "club_sandwich": {"calories": 250, "protein": 13.0, "carbs": 22.0, "fat": 12.0},
    "crab_cakes": {"calories": 220, "protein": 16.0, "carbs": 12.0, "fat": 12.0},
    "creme_brulee": {"calories": 340, "protein": 4.0, "carbs": 26.0, "fat": 24.0},
    "croque_madame": {"calories": 290, "protein": 15.0, "carbs": 20.0, "fat": 16.0},
    "cup_cakes": {"calories": 380, "protein": 3.5, "carbs": 56.0, "fat": 16.0},
    "deviled_eggs": {"calories": 200, "protein": 10.0, "carbs": 2.0, "fat": 16.0},
    "donuts": {"calories": 452, "protein": 4.9, "carbs": 51.0, "fat": 25.0},
    "dumplings": {"calories": 210, "protein": 8.0, "carbs": 28.0, "fat": 7.0},
    "edamame": {"calories": 122, "protein": 11.0, "carbs": 10.0, "fat": 5.0},
    "eggs_benedict": {"calories": 260, "protein": 12.0, "carbs": 15.0, "fat": 17.0},
    "escargots": {"calories": 220, "protein": 16.0, "carbs": 3.0, "fat": 16.0},
    "falafel": {"calories": 330, "protein": 13.0, "carbs": 32.0, "fat": 18.0},
    "filet_mignon": {"calories": 267, "protein": 26.0, "carbs": 0.0, "fat": 17.0},
    "fish_and_chips": {"calories": 230, "protein": 11.0, "carbs": 21.0, "fat": 11.0},
    "foie_gras": {"calories": 460, "protein": 11.0, "carbs": 4.0, "fat": 44.0},
    "french_fries": {"calories": 312, "protein": 3.4, "carbs": 41.0, "fat": 15.0},
    "french_onion_soup": {"calories": 95, "protein": 4.0, "carbs": 9.0, "fat": 5.0},
    "french_toast": {"calories": 230, "protein": 7.0, "carbs": 30.0, "fat": 9.0},
    "fried_calamari": {"calories": 280, "protein": 15.0, "carbs": 20.0, "fat": 15.0},
    "fried_rice": {"calories": 163, "protein": 3.0, "carbs": 31.0, "fat": 3.0},
    "frozen_yogurt": {"calories": 160, "protein": 4.0, "carbs": 24.0, "fat": 5.0},
    "garlic_bread": {"calories": 350, "protein": 8.0, "carbs": 45.0, "fat": 16.0},
    "gnocchi": {"calories": 175, "protein": 4.0, "carbs": 36.0, "fat": 1.5},
    "greek_salad": {"calories": 87, "protein": 3.0, "carbs": 6.0, "fat": 6.0},
    "grilled_cheese_sandwich": {"calories": 370, "protein": 14.0, "carbs": 30.0, "fat": 22.0},
    "grilled_salmon": {"calories": 206, "protein": 22.0, "carbs": 0.0, "fat": 12.0},
    "guacamole": {"calories": 160, "protein": 2.0, "carbs": 9.0, "fat": 15.0},
    "gyoza": {"calories": 200, "protein": 8.0, "carbs": 26.0, "fat": 7.0},
    "hamburger": {"calories": 295, "protein": 17.0, "carbs": 30.0, "fat": 14.0},
    "hot_and_sour_soup": {"calories": 65, "protein": 4.0, "carbs": 6.0, "fat": 3.0},
    "hot_dog": {"calories": 290, "protein": 10.0, "carbs": 24.0, "fat": 17.0},
    "huevos_rancheros": {"calories": 210, "protein": 10.0, "carbs": 18.0, "fat": 11.0},
    "hummus": {"calories": 166, "protein": 7.9, "carbs": 14.0, "fat": 9.6},
    "ice_cream": {"calories": 207, "protein": 3.5, "carbs": 24.0, "fat": 11.0},
    "lasagna": {"calories": 135, "protein": 8.0, "carbs": 14.0, "fat": 6.0},
    "lobster_bisque": {"calories": 160, "protein": 6.0, "carbs": 10.0, "fat": 11.0},
    "lobster_roll_sandwich": {"calories": 270, "protein": 16.0, "carbs": 24.0, "fat": 12.0},
    "macaroni_and_cheese": {"calories": 370, "protein": 14.0, "carbs": 40.0, "fat": 17.0},
    "macarons": {"calories": 440, "protein": 6.0, "carbs": 60.0, "fat": 20.0},
    "miso_soup": {"calories": 40, "protein": 3.0, "carbs": 5.0, "fat": 1.2},
    "mussels": {"calories": 172, "protein": 24.0, "carbs": 7.0, "fat": 4.5},
    "nachos": {"calories": 306, "protein": 8.0, "carbs": 36.0, "fat": 15.0},
    "omelette": {"calories": 154, "protein": 11.0, "carbs": 1.0, "fat": 11.0},
    "onion_rings": {"calories": 410, "protein": 4.5, "carbs": 44.0, "fat": 24.0},
    "oysters": {"calories": 80, "protein": 9.0, "carbs": 5.0, "fat": 2.5},
    "pad_thai": {"calories": 210, "protein": 8.0, "carbs": 32.0, "fat": 6.0},
    "paella": {"calories": 160, "protein": 9.0, "carbs": 22.0, "fat": 4.0},
    "pancakes": {"calories": 227, "protein": 6.0, "carbs": 28.0, "fat": 10.0},
    "panna_cotta": {"calories": 290, "protein": 3.5, "carbs": 28.0, "fat": 18.0},
    "peking_duck": {"calories": 330, "protein": 19.0, "carbs": 11.0, "fat": 23.0},
    "pho": {"calories": 110, "protein": 8.0, "carbs": 15.0, "fat": 2.5},
    "pizza": {"calories": 266, "protein": 11.0, "carbs": 33.0, "fat": 10.0},
    "pork_chop": {"calories": 230, "protein": 24.0, "carbs": 0.0, "fat": 14.0},
    "poutine": {"calories": 270, "protein": 8.0, "carbs": 30.0, "fat": 13.0},
    "prime_rib": {"calories": 380, "protein": 22.0, "carbs": 0.0, "fat": 32.0},
    "pulled_pork_sandwich": {"calories": 260, "protein": 18.0, "carbs": 28.0, "fat": 9.0},
    "ramen": {"calories": 436, "protein": 10.0, "carbs": 56.0, "fat": 17.0},
    "ravioli": {"calories": 220, "protein": 9.0, "carbs": 30.0, "fat": 7.0},
    "red_velvet_cake": {"calories": 370, "protein": 4.5, "carbs": 50.0, "fat": 17.0},
    "risotto": {"calories": 170, "protein": 4.0, "carbs": 25.0, "fat": 6.0},
    "samosa": {"calories": 262, "protein": 5.0, "carbs": 32.0, "fat": 13.0},
    "sashimi": {"calories": 130, "protein": 23.0, "carbs": 0.0, "fat": 4.0},
    "scallops": {"calories": 110, "protein": 20.0, "carbs": 3.0, "fat": 1.0},
    "seaweed_salad": {"calories": 70, "protein": 1.5, "carbs": 9.0, "fat": 3.0},
    "shrimp_and_grits": {"calories": 190, "protein": 10.0, "carbs": 18.0, "fat": 8.0},
    "spaghetti_bolognese": {"calories": 160, "protein": 8.0, "carbs": 20.0, "fat": 5.0},
    "spaghetti_carbonara": {"calories": 220, "protein": 9.0, "carbs": 24.0, "fat": 10.0},
    "spring_rolls": {"calories": 210, "protein": 6.0, "carbs": 26.0, "fat": 9.0},
    "steak": {"calories": 271, "protein": 26.0, "carbs": 0.0, "fat": 18.0},
    "strawberry_shortcake": {"calories": 320, "protein": 4.0, "carbs": 48.0, "fat": 13.0},
    "sushi": {"calories": 150, "protein": 7.0, "carbs": 30.0, "fat": 1.0},
    "tacos": {"calories": 226, "protein": 9.0, "carbs": 20.0, "fat": 12.0},
    "takoyaki": {"calories": 210, "protein": 7.0, "carbs": 25.0, "fat": 9.0},
    "tiramisu": {"calories": 340, "protein": 5.0, "carbs": 38.0, "fat": 18.0},
    "tuna_tartare": {"calories": 150, "protein": 22.0, "carbs": 2.0, "fat": 6.0},
    "waffles": {"calories": 291, "protein": 7.9, "carbs": 32.0, "fat": 14.0}
}


def format_food_name(food_str):
    """Converts snake_case food class names to Title Case."""
    if not food_str:
        return "Unknown"
    return food_str.replace("_", " ").title()


def predict_food(image_file):
    """
    Classifies a food image using the MobileNetV2 deep learning model.
    Applies single-pass normalization, OOD / Unknown food thresholding,
    top 3 predictions extraction, and nutrition lookup.
    """
    global model, FOOD_CLASSES

    # Reload model if loaded dynamically after training
    if model is None and os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            print("Model dynamically loaded.")
        except Exception as e:
            print("Dynamic load failed:", e)

    if len(FOOD_CLASSES) == 0 and os.path.exists(CLASSES_PATH):
        with open(CLASSES_PATH, "r", encoding="utf-8") as f:
            FOOD_CLASSES = [line.strip() for line in f if line.strip()]

    if model is None or len(FOOD_CLASSES) == 0:
        raise RuntimeError("AI Model or Food Classes are not available.")

    # 1. Load & RGB conversion
    pil_img = Image.open(image_file).convert("RGB")
    orig_size = pil_img.size

    # 2. Resize to 224x224
    pil_img_resized = pil_img.resize((224, 224))
    img_array = np.array(pil_img_resized, dtype=np.float32)

    # 3. Model inference: Note that the trained Keras model contains an internal
    # normalization layer (x / 127.5 - 1.0). Therefore, pass raw float32 [0, 255]
    # to avoid double normalization.
    img_batch = np.expand_dims(img_array, axis=0)

    # 4. Predict probabilities across 101 classes
    predictions = model.predict(img_batch, verbose=0)[0]

    # 5. Top 3 predictions
    top_indices = np.argsort(predictions)[-3:][::-1]
    predicted_index = int(top_indices[0])
    max_prob = float(predictions[predicted_index])
    confidence_pct = max_prob * 100.0

    predicted_food_raw = FOOD_CLASSES[predicted_index] if predicted_index < len(FOOD_CLASSES) else "Unknown"

    # 6. Unknown / Out-of-Distribution (OOD) Decision
    is_unknown = max_prob < CONFIDENCE_THRESHOLD

    # 7. Build Top 3 predictions list
    top_predictions = []
    for rank, idx in enumerate(top_indices, 1):
        raw_name = FOOD_CLASSES[idx] if idx < len(FOOD_CLASSES) else "Unknown"
        prob = float(predictions[idx])
        top_predictions.append({
            "rank": rank,
            "food": format_food_name(raw_name),
            "raw_class": raw_name,
            "confidence": f"{prob * 100.0:.2f}%",
            "probability": round(prob, 4)
        })

    # 8. Backend Debug Logging
    print("\n" + "=" * 60)
    print("NUTRIFIT AI PREDICTION LOG")
    print("=" * 60)
    print(f"Received Image Size: {orig_size}")
    print(f"Model Input Shape:   {img_batch.shape}")
    print(f"Predicted Index:     {predicted_index}")
    print(f"Predicted Raw Class: {predicted_food_raw}")
    print(f"Confidence:          {confidence_pct:.2f}%")
    print(f"Unknown / OOD:       {is_unknown} (Threshold: {CONFIDENCE_THRESHOLD * 100.0:.0f}%)")
    print("Top 3 Predictions:")
    for item in top_predictions:
        print(f"  {item['rank']}. {item['food']} — {item['confidence']}")

    if is_unknown:
        print("Result: Food not recognized with sufficient confidence.")
        print("=" * 60 + "\n")
        return {
            "food": "Unknown",
            "confidence": f"{confidence_pct:.2f}%",
            "calories": 0,
            "protein": 0.0,
            "carbs": 0.0,
            "fat": 0.0,
            "is_unknown": True,
            "top_predictions": top_predictions
        }

    # 9. Nutrition per 100g lookup
    food_key = predicted_food_raw.lower()
    nutrition = NUTRITION_DB.get(
        food_key,
        {"calories": 0, "protein": 0.0, "carbs": 0.0, "fat": 0.0}
    )

    formatted_name = format_food_name(predicted_food_raw)

    print(f"Nutrition (per 100g): {nutrition['calories']} kcal | P: {nutrition['protein']}g | C: {nutrition['carbs']}g | F: {nutrition['fat']}g")
    print("=" * 60 + "\n")

    return {
        "food": formatted_name,
        "confidence": f"{confidence_pct:.2f}%",
        "calories": int(round(nutrition["calories"])),
        "protein": round(float(nutrition["protein"]), 1),
        "carbs": round(float(nutrition["carbs"]), 1),
        "fat": round(float(nutrition["fat"]), 1),
        "is_unknown": False,
        "top_predictions": top_predictions
    }