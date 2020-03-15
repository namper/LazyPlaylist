import random
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


if __name__ == '__main__':
    df = pd.read_csv('data/data_v1.csv')
    df['vector'] = df['vector'].map(lambda x: np.fromstring(x, sep=';'))
    z = np.array([i for i in df['vector']])

    K = 2
    model = KMeans(n_clusters=K)
    y_km = model.fit_predict(z)

    r = (lambda: random.randint(0, 255))
    for i in range(K):
        plt.scatter(z[y_km == i, 0], z[y_km == i, 1], s=100, c='#%02X%02X%02X' % (r(), r(), r()))
    plt.show()
