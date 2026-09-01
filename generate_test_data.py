import numpy as np


def create_dummy_data():
    X = np.random.randn(100, 10).astype(np.float32)
    np.save("test_ktheory_data.npy", X)
    print("Created test_ktheory_data.npy")


if __name__ == "__main__":
    create_dummy_data()
