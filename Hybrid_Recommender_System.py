
#############################################
# PROJE: Hybrid Recommender System
#############################################

# ID'si verilen kullanıcı için item-based ve user-based recomennder yöntemlerini kullanarak tahmin yapınız.
# 5 öneri user-based modelden 5 öneri de item-based modelden ele alınız ve nihai olarak 10 öneriyi 2 modelden yapınız.

#############################################
# Görev 1: Verinin Hazırlanması
#############################################
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

# Adım 1: Movie ve Rating veri setlerini okutunuz.
# movieId, film adı ve filmin tür bilgilerini içeren veri seti
movie = pd.read_csv("data/movie.csv")

# UserID, film adı, filme verilen oy ve zaman bilgisini içeren veri seti
rating = pd.read_csv("data/rating.csv")


# Adım 2: Rating veri setine filmlerin isimlerini ve türünü movie film setini kullanarak ekleyiniz.
# Ratingdeki kullanıcıların oy kullandıkları filmlerin sadece id'si var.
# Idlere ait film isimlerini ve türünü movie veri setinden ekliyoruz.
df = pd.merge(rating, movie, on='movieId')
df.head()

# Adım 3: Herbir film için toplam kaç kişinin oy kullandığını hesaplayınız.Toplam oy kullanılma sayısı 1000'un altında olan filmleri veri setinden çıkarınız.
# Herbir film için toplam kaç kişinin oy kullanıldığını hesaplıyoruz.
# Toplam oy kullanılma sayısı 1000'in altında olan filmlerin isimlerini rare_movies de tutuyoruz.
# Ve veri setinden çıkartıyoruz
movie_counts = df['title'].value_counts()
popular_movies = movie_counts[movie_counts >= 1000].index
filtered_df = df[df['title'].isin(popular_movies)]

# Adım 4: # index'te userID'lerin sutunlarda film isimlerinin ve değer olarakta ratinglerin bulunduğu
# dataframe için pivot table oluşturunuz.
user_movie_df = filtered_df.pivot_table(
    index='userId',
    columns='title',
    values='rating'
)

# Adım 5: Yukarıda yapılan tüm işlemleri fonksiyonlaştıralım
def prepare_data(movie_path, rating_path):
    movie = pd.read_csv(movie_path)
    rating = pd.read_csv(rating_path)

    merged_df = pd.merge(rating, movie, on='movieId')

    movie_counts = merged_df['title'].value_counts()
    popular_movies = movie_counts[movie_counts >= 1000].index
    filtered_df = merged_df[merged_df['title'].isin(popular_movies)]

    user_movie_df = filtered_df.pivot_table(
        index='userId',
        columns='title',
        values='rating'
    )

    return user_movie_df



# Fonksiyonu kullanarak veri hazırlama
movie_file_path = "data/movie.csv"
rating_file_path = "data/rating.csv"

user_movie_df = prepare_data(movie_file_path, rating_file_path)


#############################################
# Görev 2: Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi
#############################################
import numpy as np

# Adım 1: Rastgele bir kullanıcı id'si seçiniz.
random_user_id = np.random.choice(user_movie_df.index)

# Eklenmesi gerek.
watched_movie_ids = df[df["userId"] == random_user_id]["movieId"].unique()

# Adım 2: Seçilen kullanıcıya ait gözlem birimlerinden oluşan random_user_df adında yeni bir dataframe oluşturunuz.
random_user_df = user_movie_df.loc[[random_user_id]]

# Adım 3: Seçilen kullanıcının oy kullandığı filmleri movies_watched adında bir listeye atayınız.
movies_watched = random_user_df.columns[
    random_user_df.notna().iloc[0]
].tolist()

print(f\"Seçilen Kullanıcı ID: {random_user_id}\")
print(\"Kullanıcının İzlediği Filmler:\")
print(movies_watched)

#############################################
# Görev 3: Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişmek
#############################################

# Adım 1: Seçilen kullanıcının izlediği fimlere ait sutunları user_movie_df'ten seçiniz ve movies_watched_df adında yeni bir dataframe oluşturuyoruz.
movies_watched_df = user_movie_df.loc[:, movies_watched]

# Adım 2: Herbir kullancının seçili user'in izlediği filmlerin kaçını izlediği bilgisini taşıyan user_movie_count adında yeni bir dataframe oluşturunuz.
# Ve yeni bir df oluşturuyoruz.
user_movie_count = movies_watched_df.notna().sum(axis=1).reset_index()
user_movie_count.columns = ['userId', 'movies_watched_count']

# Adım 3: Seçilen kullanıcının oy verdiği filmlerin yüzde 60 ve üstünü izleyenleri benzer kullanıcılar olarak görüyoruz.
# Bu kullanıcıların id’lerinden users_same_movies adında bir liste oluşturunuz.
threshold = max(len(movies_watched) * 0.60, 1)
users_same_movies = user_movie_count[user_movie_count['movies_watched_count'] >= threshold]['userId'].tolist()


#############################################
# Görev 4: Öneri Yapılacak Kullanıcı ile En Benzer Kullanıcıların Belirlenmesi
#############################################

# Adım 1: user_same_movies listesi içerisindeki seçili user ile benzerlik gösteren kullanıcıların id’lerinin bulunacağı şekilde movies_watched_df dataframe’ini filtreleyiniz.
filtered_users_df = movies_watched_df[movies_watched_df.index.isin(users_same_movies)]

# Adım 2: Kullanıcıların birbirleri ile olan korelasyonlarının bulunacağı yeni bir corr_df dataframe’i oluşturunuz.
corr_df = filtered_users_df.T.corr()

#corr_df[corr_df["user_id_1"] == random_user]



# Adım 3: Seçili kullanıcı ile yüksek korelasyona sahip (0.65’in üzerinde olan) kullanıcıları filtreleyerek top_users adında yeni bir dataframe oluşturunuz.
high_corr_users = corr_df[random_user_id][
    (corr_df[random_user_id] > 0.65) &
    (corr_df.index != random_user_id)
].index
top_users = corr_df.loc[high_corr_users]

# Adım 4:  top_users dataframe’ine rating veri seti ile merge ediniz
top_users_df = pd.merge(
    pd.DataFrame({'userId': high_corr_users}),
    rating,
    on='userId'
)

top_users_df.head()


#############################################
# Görev 5: Weighted Average Recommendation Score'un Hesaplanması ve İlk 5 Filmin Tutulması
#############################################

# Adım 1: Her bir kullanıcının corr ve rating değerlerinin çarpımından oluşan weighted_rating adında yeni bir değişken oluşturunuz.
top_users_df['correlation'] = top_users_df['userId'].map(lambda x: corr_df.loc[random_user_id, x])
top_users_df['weighted_rating'] = top_users_df['correlation'] * top_users_df['rating']

# Adım 2: Film id’sive her bir filme ait tüm kullanıcıların weighted rating’lerinin ortalama değerini içeren recommendation_df adında yeni bir
# dataframe oluşturunuz.
recommendation_df = top_users_df.groupby('movieId').agg({'weighted_rating': 'mean'}).reset_index()

# Adım 3: Adım3: recommendation_df içerisinde weighted rating'i 3.5'ten büyük olan filmleri seçiniz ve weighted rating’e göre sıralayınız.
# İlk 5 gözlemi movies_to_be_recommend olarak kaydediniz.
recommended_movies = recommendation_df[
    (recommendation_df['weighted_rating'] > 3.5) &
    (~recommendation_df['movieId'].isin(watched_movie_ids))
].sort_values('weighted_rating', ascending=False).head(5)
recommended_movies = recommended_movies.sort_values('weighted_rating', ascending=False)

# Adım 4:  Tavsiye edilen 5 filmin isimlerini getiriniz.
recommended_movies_user_based = recommended_movies.merge(
    movie[['movieId', 'title']],
    on='movieId',
    how='left'
)

#############################################
# Adım 6: Item-Based Recommendation
#############################################

def item_based_recommender(movie_name, user_movie_df, top_n=5):
    movie_series = user_movie_df[movie_name]
    correlations = user_movie_df.corrwith(movie_series).dropna()
    correlations = correlations.sort_values(ascending=False)
    return correlations[1:top_n+1].index.tolist()

# Kullanıcının en son izlediği ve en yüksek puan verdiği filmin adına göre item-based öneri yapınız.
user = 108170


# Adım 1: movie,rating veri setlerini okutunuz.

# Adım 2: Öneri yapılacak kullanıcının 5 puan verdiği filmlerden puanı en güncel olan filmin id'sinin alınız.
selected_movie_id = (
    rating[(rating["userId"] == user) & (rating["rating"] == 5)]
    .sort_values("timestamp", ascending=False)
    .iloc[0]["movieId"]
)

# Adım 3 :User based recommendation bölümünde oluşturulan user_movie_df dataframe’ini seçilen film id’sine göre filtreleyiniz.
selected_movie_title = movie.loc[
    movie["movieId"] == selected_movie_id, "title"
].values[0]

# Adım 4: Filtrelenen dataframe’i kullanarak seçili filmle diğer filmlerin korelasyonunu bulunuz ve sıralayınız.
selected_movie_ratings = user_movie_df[selected_movie_title]
correlation_with_selected_movie = user_movie_df.corrwith(selected_movie_ratings)
correlation_df = correlation_with_selected_movie.dropna().sort_values(ascending=False).to_frame()
correlation_df.columns = ['correlation']


# Adım 5: Seçili film’in kendisi haricinde ilk 5 film’I öneri olarak veriniz.
movies_from_item_based = item_based_recommender(
    selected_movie_title,
    user_movie_df,
    5
)

# İlk 5 filmi öneri olarak verin (0'da filmin kendisi var, onu dışarda bırakın)
recommended_movies_item_based = movies_from_item_based

hybrid_recommendations = (
    list(recommended_movies_user_based["title"])
    + recommended_movies_item_based
)

# Önerilen filmleri yazdır
print(f"Kullanıcı ID: {user}")
print(f"En son izlediği ve en yüksek puan verdiği film: {selected_movie_title}")
print("Önerilen Filmler:")
print(hybrid_recommendations)



