# Gözetimsiz Öğrenme ile Müşteri Segmentasyonu

#İş Problemi

# FLO müşterilerini segmentlere ayırıp bu segmentlere göre pazarlama stratejileri belirlemek istiyor.
# Buna yönelik olarak müşterilerin davranışları tanımlanacak ve bu davranışlardaki öbeklenmelere göre gruplar oluşturulacak.


# Veri Seti Hikayesi

# Veri seti Flo’dan son alışverişlerini 2020 - 2021 yıllarında OmniChannel (hem online hem offline alışveriş yapan)
# olarak yapan müşterilerin geçmiş alışveriş davranışlarından elde edilen bilgilerden oluşmaktadır.

# master_id: Eşsiz müşteri numarası
# order_channel: Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile)
# last_order_channel: En son alışverişin yapıldığı kanal
# first_order_date: Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date: Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online: Müşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline: Müşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online: Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline: Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline: Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online: Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12: Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi

# Görev 1: Veriyi Hazırlama
# Adım 1: flo_data_20K.csv verisini okutunuz.
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

pd.options.display.max_columns=100
pd.options.display.max_rows=20
pd.options.display.width=1000


df = pd.read_csv("week_7_8_9/week_9/flo_data_20k.csv")
df.head()
df.info()
df.isnull().sum()


date_cols = df.columns[df.columns.str.contains("date")]
df[date_cols] = df[date_cols].apply(pd.to_datetime)
df.dtypes
#veya

for i in df.columns:
    if "date" in i:
        df[i] = pd.to_datetime(df[i])

# Adım 2: Müşterileri segmentlerken kullanacağınız değişkenleri seçiniz.
# Not: Tenure (Müşterinin yaşı), Recency (en son kaç gün önce alışveriş yaptığı) gibi yeni değişkenler oluşturabilirsiniz

df["Frequency"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["Monetary"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

today_date = df["last_order_date"].max() + pd.Timedelta(days=2)

df["Recency"] = (today_date - df["last_order_date"]).dt.days
df["T"] = (today_date - df["first_order_date"]).dt.days

new_df = df[["Frequency","Monetary","Recency","T"]].copy()


# Görev 2: K-Means ile Müşteri Segmentasyonu

# Adım 1: Değişkenleri standartlaştırınız.
def outliner_detector(df, cols, take_care_outliners=False, print_outliners=False, q_1=0.25, q_3=0.75):
    temp = pd.DataFrame()
    data = df.copy()
    for col in cols:
        q1 = data[col].quantile(q_1)
        q3 = data[col].quantile(q_3)
        IQR = q3 - q1
        up = q3 + 1.5 * IQR
        low = q1 - 1.5 * IQR
        temp.loc[col, "Min"] = round(data[col].min())
        temp.loc[col, "Low_Limit"] = round(low)
        temp.loc[col, "Mean"] = round(data[col].mean())
        temp.loc[col, "Median"] = round(data[col].median())
        temp.loc[col,"Up_Limit"] = up
        temp.loc[col, "Max"] = data[col].max()
        temp.loc[col, "Outliner"] = "Min-Max-Outliner" if (data[col].max() > up) & (low > data[col].min())\
                                    else ("Max-Outliner" if data[col].max() > up \
                                    else ("Min-Outliner" if low > data[col].min() \
                                    else "No"))
        if take_care_outliners:
            data.loc[data[col] > up,col] = round(up)
            data.loc[data[col] < low,col] = round(low)
    if take_care_outliners:
        if print_outliners: return temp
        return data
    if print_outliners: return temp
outliner_detector(new_df, new_df.columns, print_outliners=True, q_1=0.01, q_3=0.99)

new_df = outliner_detector(new_df, new_df.columns, take_care_outliners=True, q_1=0.01, q_3=0.99)


from sklearn.preprocessing import StandardScaler
cols = new_df.columns

new_df[cols] = StandardScaler().fit_transform(new_df[cols])

# Adım 2: Optimum küme sayısını belirleyiniz.
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

elbow = KElbowVisualizer(KMeans(random_state=42), k=(2,10))
elbow.fit(new_df)
elbow.show()

elbow.elbow_value_



# Adım 3: Modelinizi oluşturunuz ve müşterilerinizi segmentleyiniz.
kmeans = KMeans(n_clusters=elbow.elbow_value_, random_state=42).fit(new_df)
kmeans.labels_

df["Segment"] = kmeans.labels_ + 1


# Adım 4: Herbir segmenti istatistiksel olarak inceleyeniz
df.groupby("Segment").agg({"Recency":["count","min","max","mean"],
                           "T": ["min","max","mean"],
                           "Frequency":["min","max","mean"],
                           "Monetary":["min","max","mean"]})

df.groupby("Segment").agg(Count=("Segment","count"),
                          Recency_Mean=("Recency","mean"),
                          T_Mean=("T","mean"),
                          Frequency_Mean=("Frequency","mean"),
                          Monetary_Mean=("Monetary","mean"))

print(df.groupby("Segment").agg(Count=("Segment","count"),
                          Recency=("Recency", lambda x: f" Mean: {x.mean():.0f}, Max:{x.max()}, Min:{x.min()}"),
                          T=("T", lambda x: f" Mean: {x.mean():.0f}, Max:{x.max()}, Min:{x.min()}"),
                          Frequency=("Frequency", lambda x: f" Mean: {x.mean():.0f}, Max:{x.max()}, Min:{x.min()}"),
                          Monetary=("Monetary", lambda x: f" Mean: {x.mean():.0f}, Max:{x.max():.0f}, Min:{x.min():.0f}")).to_markdown())



# Görev 3: Hierarchical Clustering ile Müşteri Segmentasyonu
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# Adım 1: Görev 2'de standırlaştırdığınız dataframe'i kullanarak optimum küme sayısını belirleyiniz.

hc = linkage(new_df, "ward")


dendrogram(hc,
           truncate_mode="lastp", #Dendrogramın nasıl kırpılacağını belirten bir parametredir. lastp seçeneği, en son p kümesin gösterileceği anlamına gelir.
           p=10, # gösterilecek son p kümenin sayısını belirtir.
           show_contracted=False, # Kümeleme düğümlerinin birleştirildiğini gösteren bir parametre
           leaf_font_size=13) #Yaprak düğümlerinin yazı font boyutu
plt.axhline(y=100, color='r', linestyle='--')
plt.show()


# Adım 2: Modelinizi oluşturunuz ve müşterileriniz segmentleyiniz.

df["Hc_Segments"] = fcluster(hc, 100, criterion="distance")
# Adım 3: Her bir segmenti istatistiksel olarak inceleyeniz

print(df.groupby("Hc_Segments").agg(Count=("Hc_Segments","count"),
                          Recency=("Recency", lambda x: f" Mean: {x.mean():.0f}, Max:{x.max()}, Min:{x.min()}"),
                          T=("T", lambda x: f" Mean: {x.mean():.0f}, Max:{x.max()}, Min:{x.min()}"),
                          Frequency=("Frequency", lambda x: f" Mean: {x.mean():.0f}, Max:{x.max()}, Min:{x.min()}"),
                          Monetary=("Monetary", lambda x: f" Mean: {x.mean():.0f}, Max:{x.max():.0f}, Min:{x.min():.0f}")).to_markdown())



# Bonus:  Agglomerative Clustering ile Müşteri Segmentasyonu
from sklearn.cluster import AgglomerativeClustering

# Adım 1: Görev 2'de standırlaştırdığınız dataframe'i kullanarak optimum küme sayısını belirleyiniz.
elbow = KElbowVisualizer(KMeans(random_state=42), k=(2,10))
elbow.fit(new_df)
elbow.show()

elbow.elbow_value_
# Adım 2: Modelinizi oluşturunuz ve müşterileriniz segmentleyiniz.
cluster = AgglomerativeClustering(n_clusters=5, linkage="ward")

clusters = cluster.fit_predict(new_df)

df["Agg_Segments"] = clusters + 1
# Adım 3: Her bir segmenti istatistiksel olarak inceleyeniz
print(df.groupby("Agg_Segments").agg(Count=("Agg_Segments","count"),
                          Recency=("Recency", lambda x: f" Mean: {x.mean():.0f}, Max:{x.max()}, Min:{x.min()}"),
                          T=("T", lambda x: f" Mean: {x.mean():.0f}, Max:{x.max()}, Min:{x.min()}"),
                          Frequency=("Frequency", lambda x: f" Mean: {x.mean():.0f}, Max:{x.max()}, Min:{x.min()}"),
                          Monetary=("Monetary", lambda x: f" Mean: {x.mean():.0f}, Max:{x.max():.0f}, Min:{x.min():.0f}")).to_markdown())

