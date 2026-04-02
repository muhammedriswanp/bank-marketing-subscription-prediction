import joblib
import os


def save_model(model, filename: str, folder: str = 'models') -> None:
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    joblib.dump(model, path)
    print(f"Model saved → {path}")


def load_model(filename: str, folder: str = 'models'):
    path = os.path.join(folder, filename)
    return joblib.load(path)