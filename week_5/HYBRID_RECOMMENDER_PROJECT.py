
#############################################
# PROJE: Hybrid Recommender System
#############################################

# ID'si verilen kullanıcı için item-based ve user-based recomennder yöntemlerini kullanarak tahmin yapınız.
# 5 öneri user-based modelden 5 öneri de item-based modelden ele alınız ve nihai olarak 10 öneriyi 2 modelden yapınız.

#############################################
# Görev 1: Verinin Hazırlanması
#############################################
import pandas as pd
pd.options.display.max_columns=10
pd.options.display.max_rows=20
pd.options.display.float_format = '{:.3f}'.format
pd.options.display.width = 1000

# Adım 1: Movie ve Rating veri setlerini okutunuz.
# movieId, film adı ve filmin tür bilgilerini içeren veri seti
movie = pd.read_csv("week_5/datasets/movie.csv")

# UserID, film adı, filme verilen oy ve zaman bilgisini içeren veri seti
rating = pd.read_csv("week_5/datasets/rating.csv")

# Adım 2: Rating veri setine filmlerin isimlerini ve türünü movie film setini kullanarak ekleyiniz.
# Ratingdeki kullanıcıların oy kullandıkları filmlerin sadece id'si var.
# Idlere ait film isimlerini ve türünü movie veri setinden ekliyoruz.

df = pd.merge(movie,rating, how="inner", on="movieId")
df.isnull().sum()

vahit_hoca_df = movie.merge(rating, how="left", on="movieId")
vahit_hoca_df.isnull().sum()
vahit_hoca_df[vahit_hoca_df["userId"].isna()]

# Adım 3: Herbir film için toplam kaç kişinin oy kullandığını hesaplayınız.Toplam oy kullanılma sayısı 1000'un altında olan filmleri veri setinden çıkarınız.
# Herbir film için toplam kaç kişinin oy kullanıldığını hesaplıyoruz.
# Toplam oy kullanılma sayısı 1000'in altında olan filmlerin isimlerini rare_movies de tutuyoruz.
# Ve veri setinden çıkartıyoruz
values_pd = df["title"].value_counts()

rare_movies = values_pd[values_pd < 1000].index

df_ = df[~df["title"].isin(rare_movies)]

# Adım 4: # index'te userID'lerin sutunlarda film isimlerinin ve değer olarakta ratinglerin bulunduğu
# dataframe için pivot table oluşturunuz. (adını user_movie_df koyun)

user_movie_df = df_.pivot_table(index="userId",columns="title",values="rating")
#df.groupby(["user_id","title"]).agg({"rating":"mean"}).unstack()
user_movie_df.shape
user_movie_df.head(20)
# Adım 5: Yukarıda yapılan tüm işlemleri fonksiyonlaştıralım

def yuk(df1, df2, min_vote=1000):
    df1,df2 = df1.copy(), df2.copy()
    merged = pd.merge(df1, df2, how="inner", on="movieId")
    counter = merged["title"].value_counts()
    rares = counter[counter < min_vote].index
    merged_non_rares = merged[~merged["title"].isin(rares)]
    pivottable = merged_non_rares.pivot_table(values="rating", index="userId", columns="title")
    return pivottable

#############################################
# Görev 2: Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi
#############################################

# Adım 1: Rastgele bir kullanıcı id'si seçiniz.

lucky_guy = user_movie_df.sample(1,random_state=45).index[0]


# Adım 2: Seçilen kullanıcıya ait gözlem birimlerinden oluşan random_user_df adında yeni bir dataframe oluşturunuz.

random_user_df = user_movie_df[user_movie_df.index==lucky_guy]

# Adım 3: Seçilen kullanıcının oy kullandığı filmleri movies_watched adında bir listeye atayınız.

#movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist() #hocanınki

movies_watched = random_user_df.dropna(axis=1).columns.tolist()
lucky_guy_watched_ids = df_["movieId"][(df_["userId"]==lucky_guy)]
#############################################
# Görev 3: Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişmek
#############################################

# Adım 1: Seçilen kullanıcının izlediği fimlere ait sutunları user_movie_df'ten seçiniz ve movies_watched_df adında yeni bir dataframe oluşturuyoruz.

movies_watched_df = user_movie_df[movies_watched]

# Adım 2: Herbir kullancının seçili user'in izlediği filmlerin kaçını izlediği bilgisini taşıyan user_movie_count adında yeni bir dataframe oluşturunuz.

#user_movie_count = movies_watched_df.T.notnull().sum()

user_movie_count = movies_watched_df.notnull().sum(axis=1)
user_movie_count.sort_values(ascending=False).head(50)
user_movie_count.max()
# Adım 3: Seçilen kullanıcının oy verdiği filmlerin yüzde 60 ve üstünü izleyenleri benzer kullanıcılar olarak görüyoruz.
# Bu kullanıcıların id’lerinden users_same_movies adında bir liste oluşturunuz.


users_same_movies = user_movie_count[user_movie_count > (movies_watched_df.shape[1] * 60 ) / 100].index

#############################################
# Görev 4: Öneri Yapılacak Kullanıcı ile En Benzer Kullanıcıların Belirlenmesi
#############################################

# Adım 1: user_same_movies listesi içerisindeki seçili user ile benzerlik gösteren kullanıcıların id’lerinin bulunacağı şekilde movies_watched_df dataframe’ini filtreleyiniz.

filted_df = movies_watched_df[movies_watched_df.index.isin(users_same_movies)]

# Adım 2: Kullanıcıların birbirleri ile olan korelasyonlarının bulunacağı yeni bir corr_df dataframe’i oluşturunuz.
#corr_df[corr_df["user_id_1"] == random_user]

corr_df = filted_df.T.corr().unstack()
    # corr_df = pd.DataFrame(corr_df, columns=["corr"])
    #
    # corr_df.index.names = ['user_id_1', 'user_id_2']
    #
    # corr_df = corr_df.reset_index()
corr_df[lucky_guy].sort_values(ascending=False)
# Adım 3: Seçili kullanıcı ile yüksek korelasyona sahip (0.65’in üzerinde olan) kullanıcıları filtreleyerek top_users adında yeni bir dataframe oluşturunuz.
top_users = pd.DataFrame(corr_df[lucky_guy][corr_df[lucky_guy] > 0.65], columns=["corr"])
top_users
    #top_users = corr_df[(corr_df["user_id_1"] == lucky_guy) & (corr_df["corr"] >= 0.65)][["user_id_2", "corr"]].reset_index(drop=True)

# Adım 4:  top_users dataframe’ine rating veri seti ile merge ediniz
    #top_users.rename(columns={"user_id_2": "userId"}, inplace=True)

top_users_ratings = pd.merge(top_users, rating[["userId", "movieId", "rating"]], how='inner', on="userId")

top_users_ratings = top_users_ratings[~(top_users_ratings["movieId"].isin(lucky_guy_watched_ids))]

#############################################
# Görev 5: Weighted Average Recommendation Score'un Hesaplanması ve İlk 5 Filmin Tutulması
#############################################

# Adım 1: Her bir kullanıcının corr ve rating değerlerinin çarpımından oluşan weighted_rating adında yeni bir değişken oluşturunuz.

top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']

top_users_ratings.sort_values(by="corr",ascending=False)
# Adım 2: Film id’si ve her bir filme ait tüm kullanıcıların weighted rating’lerinin ortalama değerini içeren recommendation_df adında yeni bir
# dataframe oluşturunuz.

recommendation_df = top_users_ratings.pivot_table(values="weighted_rating", index="movieId", aggfunc="mean")
recommendation_df.sort_values(by="weighted_rating" ,ascending=False)

# Adım 3: Adım3: recommendation_df içerisinde weighted rating'i 3.5'ten büyük olan filmleri seçiniz ve weighted rating’e göre sıralayınız.
# İlk 5 gözlemi movies_to_be_recommend olarak kaydediniz.

movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.5].sort_values(by="weighted_rating", ascending=False).head(5)

# Adım 4:  Tavsiye edilen 5 filmin isimlerini getiriniz.

pd.merge(movies_to_be_recommend,movie,how="inner",on="movieId")[["movieId","weighted_rating","title"]]

#############################################
# Adım 6: Item-Based Recommendation
#############################################

# Kullanıcının en son izlediği ve en yüksek puan verdiği filmin adına göre item-based öneri yapınız.
lucky_guy

# Adım 1: movie,rating veri setlerini okutunuz.
movie,rating

# Adım 2: Öneri yapılacak kullanıcının 5 puan verdiği filmlerden puanı en güncel olan filmin id'sinin alınız.

pick = rating[(rating["rating"] == 5) & (rating["userId"]==lucky_guy)].sort_values(by="timestamp", ascending=False).iloc[0]["movieId"]


# Adım 3 :User based recommendation bölümünde oluşturulan user_movie_df dataframe’ini seçilen film id’sine göre filtreleyiniz.
picked_movie_name = movie["title"][movie["movieId"]==pick].iloc[0]

filted = user_movie_df[picked_movie_name]

filted[filted.notna()]
# Adım 4: Filtrelenen dataframe’i kullanarak seçili filmle diğer filmlerin korelasyonunu bulunuz ve sıralayınız.
users_wo_lucky = user_movie_df.drop(lucky_guy,axis=0).drop(movies_watched,axis=1)
filted_wo_lucky = filted.drop(lucky_guy,axis=0)

movies_similarity = users_wo_lucky.corrwith(filted_wo_lucky).sort_values(ascending=False).reset_index()

movies_similarity.columns=["title","similarity"]

# Adım 5: Seçili film’in kendisi haricinde ilk 5 film’I öneri olarak veriniz.

movies_similarity.sort_values(by="similarity",ascending=False).iloc[1:6]


#SO
# Hybrid : (yoktu ben ekledim)
# Kullanıcının 5 puan verdiği filimle yüksek benzerliğe sahip filmler ile diğer benzer kullanıcıların ortalama olarak 3.5 ve üzeri puan verdiği filmleri birleştir.
# Benzerlik skorunu ve ortalama puanını çarp öner, sıralamaya göre öner

movies_ordered_by_rating = pd.merge(recommendation_df,movie,how="inner",on="movieId")[["movieId","weighted_rating","title"]]

merged = pd.merge(movies_similarity,movies_ordered_by_rating,how="inner", on="title")

merged["hybrid"] = merged["similarity"] * merged["weighted_rating"]

merged[["title","hybrid"]].sort_values(by="hybrid", ascending=False).head(10)
