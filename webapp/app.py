from flask import Flask, render_template, request
import os
import cv2
import joblib
from skimage.feature import hog

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

svm = joblib.load(os.path.join(MODEL_DIR, "svm_pca_dog_cat.pkl"))
pca = joblib.load(os.path.join(MODEL_DIR, "pca.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

imgsize = 128
MODEL_ACCURACY = 85

def extractfeature(imgpath):
    img = cv2.imread(imgpath)
    img = cv2.resize(img, (imgsize, imgsize))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    feat = hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    feat = scaler.transform([feat])
    feat = pca.transform(feat)
    return feat

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image_url = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            filename = file.filename
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(image_path)

            feat = extractfeature(image_path)
            pred = svm.predict(feat)[0]

            prediction = "Dog 🐶" if pred == 1 else "Cat 🐱"
            image_url = f"/static/uploads/{filename}"

    return render_template(
        "index.html",
        prediction=prediction,
        image=image_url,
        accuracy=MODEL_ACCURACY
    )

if __name__ == "__main__":
    app.run()
