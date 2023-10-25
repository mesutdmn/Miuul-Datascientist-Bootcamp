
###################################################
# PROJE: Rating Product & Sorting Reviews in Amazon
###################################################

###################################################
# İş Problemi
###################################################

# E-ticaretteki en önemli problemlerden bir tanesi ürünlere satış sonrası verilen puanların doğru şekilde hesaplanmasıdır.
# Bu problemin çözümü e-ticaret sitesi için daha fazla müşteri memnuniyeti sağlamak, satıcılar için ürünün öne çıkması ve satın
# alanlar için sorunsuz bir alışveriş deneyimi demektir. Bir diğer problem ise ürünlere verilen yorumların doğru bir şekilde sıralanması
# olarak karşımıza çıkmaktadır. Yanıltıcı yorumların öne çıkması ürünün satışını doğrudan etkileyeceğinden dolayı hem maddi kayıp
# hem de müşteri kaybına neden olacaktır. Bu 2 temel problemin çözümünde e-ticaret sitesi ve satıcılar satışlarını arttırırken müşteriler
# ise satın alma yolculuğunu sorunsuz olarak tamamlayacaktır.

###################################################
# Veri Seti Hikayesi
###################################################

# Amazon ürün verilerini içeren bu veri seti ürün kategorileri ile çeşitli metadataları içermektedir.
# Elektronik kategorisindeki en fazla yorum alan ürünün kullanıcı puanları ve yorumları vardır.

# Değişkenler:
# reviewerID: Kullanıcı ID’si
# asin: Ürün ID’si
# reviewerName: Kullanıcı Adı
# helpful: Faydalı değerlendirme derecesi
# reviewText: Değerlendirme
# overall: Ürün rating’i
# summary: Değerlendirme özeti
# unixReviewTime: Değerlendirme zamanı
# reviewTime: Değerlendirme zamanı Raw
# day_diff: Değerlendirmeden itibaren geçen gün sayısı
# helpful_yes: Değerlendirmenin faydalı bulunma sayısı
# total_vote: Değerlendirmeye verilen oy sayısı



###################################################
# GÖREV 1: Average Rating'i Güncel Yorumlara Göre Hesaplayınız ve Var Olan Average Rating ile Kıyaslayınız.
###################################################

# Paylaşılan veri setinde kullanıcılar bir ürüne puanlar vermiş ve yorumlar yapmıştır.
# Bu görevde amacımız verilen puanları tarihe göre ağırlıklandırarak değerlendirmek.
# İlk ortalama puan ile elde edilecek tarihe göre ağırlıklı puanın karşılaştırılması gerekmektedir.


###################################################
# Adım 1: Veri Setini Okutunuz ve Ürünün Ortalama Puanını Hesaplayınız.
###################################################
import pandas as pd
import scipy.stats as st
import math
pd.options.display.max_columns=None
pd.options.display.float_format = '{:.3f}'.format
pd.options.display.width = 1000

df_ = pd.read_csv("week_4/amazon_review.csv")
df = df_.copy()

df.head()
df.shape

print(f"Ürünün Ortalama Puanı: {df['overall'].mean():.3f}")

###################################################
# Adım 2: Tarihe Göre Ağırlıklı Puan Ortalamasını Hesaplayınız.
###################################################

weighted_rating = df['overall'][(df["day_diff"]<=30)].mean() * 28/100 + \
df['overall'][(df["day_diff"]>30) & (df["day_diff"]<=90)].mean() * 26/100 + \
df['overall'][(df["day_diff"]>90) & (df["day_diff"]<=180)].mean() * 24/100 + \
df['overall'][(df["day_diff"]>180)].mean() * 22/100

print(f"Tarihe Göre Ağırlıklı Puan Ortalamasını: {weighted_rating:.3f}")

###################################################
# Görev 2: Ürün için Ürün Detay Sayfasında Görüntülenecek 20 Review'i Belirleyiniz.
###################################################


###################################################
# Adım 1. helpful_no Değişkenini Üretiniz
###################################################

# Not:
# total_vote bir yoruma verilen toplam up-down sayısıdır.
# up, helpful demektir.
# veri setinde helpful_no değişkeni yoktur, var olan değişkenler üzerinden üretilmesi gerekmektedir.


df["helpful_no"] = df["total_vote"] - df["helpful_yes"]

###################################################
# Adım 2. score_pos_neg_diff, score_average_rating ve wilson_lower_bound Skorlarını Hesaplayıp Veriye Ekleyiniz
###################################################

def score_pos_neg_diff(col1,col2):
    return col1 - col2

    df["score_pos_neg_diff"] = score_pos_neg_diff(df["helpful_yes"], df["helpful_no"])

df["score_pos_neg_diff"] = (df["score_pos_neg_diff"] - df["score_pos_neg_diff"].min()) / (df["score_pos_neg_diff"].max() - df["score_pos_neg_diff"].min())
def score_average_rating(col1, col2):
    if col1 + col2 == 0:
        return 0
    return col1 / (col1 + col2)

df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"],
                                                                     x["helpful_no"]), axis=1)

def wilson_lower_bound(up, down, confidence=0.95):
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"],
                                                                     x["helpful_no"]), axis=1)

##################################################
# Adım 3. 20 Yorumu Belirleyiniz ve Sonuçları Yorumlayınız.
###################################################

df.sort_values(by="wilson_lower_bound", ascending=False).head(20)
