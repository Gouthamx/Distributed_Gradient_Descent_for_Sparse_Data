# approach_utils.py
def save_results(filename, data):
    import pickle
    with open(filename, "wb") as f:
        pickle.dump(data, f)

def load_results(filename):
    import pickle
    with open(filename, "rb") as f:
        return pickle.load(f)
