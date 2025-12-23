from flask import Flask, render_template, request
import os
import cv2
import numpy as np
import joblib

from skimage.feature import hog

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"

svm = joblib.load("../models/svm_pca_dog_cat.pkl")
pca = joblib.load("../models/pca.pkl")
scaler = joblib.load("../models/scaler.pkl")

imgsize = 128

def extractfeature(imgpath):
    img = cv2.imread(imgpath)
    img = cv2.resize(img, (imgsize, imgsize))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    features = hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    features = scaler.transform([features])
    features = pca.transform(features)

    return features

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image_path = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(image_path)

            feat = extractfeature(image_path)
            pred = svm.predict(feat)[0]

            prediction = "Dog 🐶" if pred == 1 else "Cat 🐱"

    return render_template("index.html", prediction=prediction, image=image_path)

if __name__ == "__main__":
    app.run(debug=True)
