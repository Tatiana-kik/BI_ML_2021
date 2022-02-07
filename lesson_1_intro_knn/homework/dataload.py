import pickle
from sklearn.datasets import fetch_openml
X, y = fetch_openml(name="Fashion-MNIST", return_X_y=True, as_frame=False)
pickle.dump(X, open("X.txt", "wb"))
pickle.dump(y, open("y.txt", "wb"))