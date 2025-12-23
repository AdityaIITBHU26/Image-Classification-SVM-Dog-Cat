from flask import Flask, render_template, request
import os
import cv2
import joblib
from skimage.feature import hog

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
HOG_FOLDER = os.path.join(BASE_DIR, "static", "hog")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HOG_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

svm = joblib.load(os.path.join(MODEL_DIR, "svm_pca_dog_cat.pkl"))
pca = joblib.load(os.path.join(MODEL_DIR, "pca.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

IMG_SIZE = 128
MODEL_ACCURACY = 85  # reported test accuracy

def extract_feature(img_path, hog_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    features, hog_img = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=True
    )

    hog_img = (hog_img - hog_img.min()) / (hog_img.max() - hog_img.min())
    hog_img = (hog_img * 255).astype("uint8")
    cv2.imwrite(hog_path, hog_img)

    features = scaler.transform([features])
    features = pca.transform(features)
    return features

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
            feat = extract_feature(image_path, hog_path)
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
