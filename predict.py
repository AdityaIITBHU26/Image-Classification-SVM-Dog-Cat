import cv2
import joblib
from skimage.feature import hog

imgsize = 128

svm = joblib.load("models/svm_pca_dog_cat.pkl")
pca = joblib.load("models/pca.pkl")
scaler = joblib.load("models/scaler.pkl")

def predict(imgpath):
    img = cv2.imread(imgpath)
    img = cv2.resize(img, (imgsize, imgsize))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    feat = hog(img, orientations=9, pixels_per_cell=(8,8),
               cells_per_block=(2,2), block_norm="L2-Hys")

    feat = scaler.transform([feat])
    feat = pca.transform(feat)

    pred = svm.predict(feat)[0]
    return "Dog" if pred == 1 else "Cat"

print(predict("sample_images/dog.jpg"))
