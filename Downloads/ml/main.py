import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import pickle

def recommend_movies_for_new_user(favorite_movies, knn_model, sparse_matrix, movie_indices, top_n=20):
    """
    Рекомендует фильмы для нового пользователя на основе KNN.
    :param favorite_movies: список ID фильмов нового пользователя.
    :param knn_model: обученная KNN модель.
    :param sparse_matrix: разреженная матрица пользователь-фильм.
    :param movie_indices: индексы фильмов в матрице.
    :param top_n: количество фильмов для рекомендаций.
    :return: список рекомендованных фильмов.
    """
    movie_id_to_index = {movie: idx for idx, movie in enumerate(movie_indices)}
    favorite_indices = [movie_id_to_index[movie] for movie in favorite_movies if movie in movie_id_to_index]

    similarity_scores = []
    for idx in favorite_indices:
        distances, indices = knn_model.kneighbors(sparse_matrix[idx], n_neighbors=top_n + len(favorite_movies))
        similarity_scores.extend(indices.flatten())

    # Подсчёт частоты появления фильмов среди соседей
    recommendation_counts = pd.Series(similarity_scores).value_counts()
    # Убираем фильмы, которые пользователь уже указал
    recommendation_counts = recommendation_counts[~recommendation_counts.index.isin(favorite_indices)]
    
    recommended_indices = recommendation_counts.nlargest(top_n).index
    recommended_movies = [movie_indices[i] for i in recommended_indices]
    return recommended_movies

with open('/content/model_52.pkl', 'rb') as file:
    knn_model = pickle.load(file)
print("Модель успешно загружена")

new_user_favorite_movies = [234, 52, 53, 10681, 12, 293310, 331781, 1430, 158999, 39452, 27205, 155, 76341, 49026, 680, 920, 2330, 77, 64682, 157336]

def prepare_data(rating_df):
    """
    Преобразует DataFrame с оценками в разреженную матрицу пользователь-фильм.
    :param rating_df: DataFrame с полями ['userId', 'movid', 'rating'].
    :return: разреженная матрица пользователь-фильм, индексы фильмов и пользователей.
    """
    user_movie_matrix = rating_df.pivot(index='movieId', columns='userId', values='rating').fillna(0)
    sparse_matrix = csr_matrix(user_movie_matrix)
    return sparse_matrix, user_movie_matrix.index, user_movie_matrix.columns


if __name__ == "__main__":
    # Загрузка данных
    rating_small = pd.read_csv('/content/ratings_small.csv')
    # Подготовка данных
    sparse_matrix, movie_indices, user_indices = prepare_data(rating_small)
    # Получение рекомендаций
    recommended_movies = recommend_movies_for_new_user(new_user_favorite_movies, knn_model, sparse_matrix, movie_indices, top_n=30)

    print("Рекомендации для нового пользователя:")
    print(recommended_movies)