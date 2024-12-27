import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import pickle

#userId	movieId	rating

# 1. Загрузка и подготовка данных
def prepare_data(rating_df):
    """
    Преобразует DataFrame с оценками в разреженную матрицу пользователь-фильм.
    :param rating_df: DataFrame с полями ['userId', 'movid', 'rating'].
    :return: разреженная матрица пользователь-фильм, индексы фильмов и пользователей.
    """
    user_movie_matrix = rating_df.pivot(index='movieId', columns='userId', values='rating').fillna(0)
    sparse_matrix = csr_matrix(user_movie_matrix)
    return sparse_matrix, user_movie_matrix.index, user_movie_matrix.columns

# 2. Обучение KNN модели
def train_knn_model(sparse_matrix, n_neighbors=10):
    """
    Обучает KNN модель на основе косинусной схожести.
    :param sparse_matrix: разреженная матрица пользователь-фильм.
    :param n_neighbors: количество ближайших соседей.
    :return: обученная KNN модель.
    """
    model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=n_neighbors)
    model.fit(sparse_matrix)

    return model

if __name__ == "__main__":
    # Загрузка данных
    rating_small = pd.read_csv('/content/ratings_small.csv')  # Должен содержать ['userid', 'movid', 'rating']

    # Подготовка данных
    sparse_matrix, movie_indices, user_indices = prepare_data(rating_small)

    # Обучение модели KNN
    knn_model = train_knn_model(sparse_matrix, n_neighbors=10)

    with open('model_02_10.pkl', 'wb') as file:
        pickle.dump(knn_model, file)
    print("Модель экспортирована в файл 'model.......pkl'")