import os
import cv2
import numpy as np
from tqdm import tqdm
from skimage.feature import hog
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

imgsize = 128
X = []
y = []

for file in tqdm(os.listdir("train")):
    label = 0 if file.startswith("cat") else 1
    path = os.path.join("train", file)

    img = cv2.imread(path)
    img = cv2.resize(img, (imgsize, imgsize))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    features = hog(img, orientations=9, pixels_per_cell=(8,8),
                   cells_per_block=(2,2), block_norm="L2-Hys")

    X.append(features)
    y.append(label)

X = np.array(X)
y = np.array(y)

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
Xtrain = scaler.fit_transform(Xtrain)
Xtest = scaler.transform(Xtest)

pca = PCA(n_components=0.95)
Xtrain = pca.fit_transform(Xtrain)
Xtest = pca.transform(Xtest)

params = {"C":[1,10], "gamma":[0.01,0.001], "kernel":["rbf"]}
grid = GridSearchCV(SVC(), params, cv=3)
grid.fit(Xtrain, ytrain)

model = grid.best_estimator_

ypred = model.predict(Xtest)
print("Accuracy:", accuracy_score(ytest, ypred))

joblib.dump(model, "models/svm_pca_dog_cat.pkl")
joblib.dump(pca, "models/pca.pkl")
joblib.dump(scaler, "models/scaler.pkl")
