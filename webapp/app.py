from flask import Flask, render_template, request
import os
import cv2
import joblib
import numpy as np
from skimage.feature import hog

# ---------------- PATH SETUP ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
HOG_FOLDER = os.path.join(BASE_DIR, "static", "hog")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HOG_FOLDER, exist_ok=True)

# ---------------- APP ----------------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ---------------- LOAD MODELS ----------------
svm = joblib.load(os.path.join(MODEL_DIR, "svm_pca_dog_cat.pkl"))
pca = joblib.load(os.path.join(MODEL_DIR, "pca.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

IMG_SIZE = 128
MODEL_ACCURACY = 85  # shown on UI only

# ---------------- FEATURE + HOG IMAGE ----------------
def extractfeature(imgpath, hog_save_path):
    img = cv2.imread(imgpath)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    features, hog_image = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=True
    )

    hog_image = (hog_image - hog_image.min()) / (hog_image.max() - hog_image.min())
    hog_image = (hog_image * 255).astype("uint8")
    cv2.imwrite(hog_save_path, hog_image)

    features = scaler.transform([features])
    features = pca.transform(features)

    return features

# ---------------- ROUTE ----------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image_url = None
    hog_url = None

    if request.method == "POST":
        file = request.files.get("image")
        if file:
            filename = file.filename

            image_path = os.path.join(UPLOAD_FOLDER, filename)
            hog_path = os.path.join(HOG_FOLDER, filename)

            file.save(image_path)

            feat = extractfeature(image_path, hog_path)
            pred = svm.predict(feat)[0]

            prediction = "Dog 🐶" if pred == 1 else "Cat 🐱"
            image_url = f"/static/uploads/{filename}"
            hog_url = f"/static/hog/{filename}"

    return render_template(
        "index.html",
        prediction=prediction,
        image=image_url,
        hog_image=hog_url,
        accuracy=MODEL_ACCURACY
    )

if __name__ == "__main__":
    app.run()
