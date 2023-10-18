
###############################################################
# RFM ile Müşteri Segmentasyonu (Customer Segmentation with RFM)
###############################################################

###############################################################
# İş Problemi (Business Problem)
###############################################################
# FLO müşterilerini segmentlere ayırıp bu segmentlere göre pazarlama stratejileri belirlemek istiyor.
# Buna yönelik olarak müşterilerin davranışları tanımlanacak ve bu davranış öbeklenmelerine göre gruplar oluşturulacak..

###############################################################
# Veri Seti Hikayesi
###############################################################

# Veri seti son alışverişlerini 2020 - 2021 yıllarında OmniChannel(hem online hem offline alışveriş yapan) olarak yapan müşterilerin geçmiş alışveriş davranışlarından
# elde edilen bilgilerden oluşmaktadır.

# master_id: Eşsiz müşteri numarası
# order_channel : Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile, Offline)
# last_order_channel : En son alışverişin yapıldığı kanal
# first_order_date : Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date : Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online : Muşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline : Muşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online : Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline : Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline : Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online : Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12 : Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi

###############################################################
# GÖREVLER
###############################################################

# GÖREV 1: Veriyi Anlama (Data Understanding) ve Hazırlama

import pandas as pd
import datetime as dt

pd.options.display.max_columns=None
pd.options.display.float_format = '{:.3f}'.format
pd.options.display.width = 1000
import warnings
warnings.filterwarnings("ignore")
# 1. flo_data_20K.csv verisini okuyunuz.
df_ = pd.read_csv("flo_data_20k.csv")
df = df_.copy()

# 2. Veri setinde

# a. İlk 10 gözlem,

df.head(10)

# b. Değişken isimleri,
df.columns

# c. Betimsel istatistik,
df.describe().T

# d. Boş değer,
df.isnull().sum()

# e. Değişken tipleri, incelemesi yapınız.
df.dtypes
df.loc[:,df.columns.str.contains("date")]
# 3. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir. Herbir müşterinin toplam  alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.
df["frequency"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["monetary"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

# 4. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
df["last_order_date_offline"] = pd.to_datetime(df["last_order_date_offline"])

df[df.columns[df.columns.str.contains("date")]] = df[df.columns[df.columns.str.contains("date")]].apply(pd.to_datetime)
df.loc[:,df.columns.str.contains("date")] = df.loc[:,df.columns.str.contains("date")].apply(pd.to_datetime)

# 5. Alışveriş kanallarındaki müşteri sayısının, ortalama alınan ürün sayısının ve ortalama harcamaların dağılımına bakınız.
# 5  Alışveriş kanallarındaki müşteri sayısının, toplam alınan ürünsayısının ve toplam harcamaların dağılımına bakınız
df.groupby(["order_channel"]).agg({"order_channel" : "count",
                                    "frequency": ["mean","sum"],
                                    "monetary": ["mean","sum"]})

df.groupby(["last_order_channel"]).agg({"last_order_channel" : "count",
                                    "frequency": ["mean","sum"],
                                    "monetary": ["mean","sum"]})
# 6. En fazla kazancı getiren ilk 10 müşteriyi sıralayınız.

df.groupby(["master_id"]).agg({"monetary":"sum"}).sort_values(by="monetary", ascending=False).head(10)

# 7. En fazla siparişi veren ilk 10 müşteriyi sıralayınız.
df.groupby(["master_id"]).agg({"frequency":"sum"}).sort_values(by="frequency", ascending=False).head(10)

# 8. Veri ön hazırlık sürecini fonksiyonlaştırınız.

def data_prep(df):
    df["frequency"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
    df["monetary"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
    df[df.columns[df.columns.str.contains("date")]] = df[df.columns[df.columns.str.contains("date")]].apply(
        pd.to_datetime)
    return df
data_prep(df)

import timeit
# GÖREV 2: RFM Metriklerinin Hesaplanması
df["last_order_date"].max()
today_date = df["last_order_date"].max() + pd.Timedelta(days=2)

%%timeit -r1 -n1
rfm = df.groupby(["master_id"]).agg({'last_order_date': lambda x: (today_date - x.max()).days,
                                        "frequency":  lambda x: x,
                                        "monetary":  lambda x: x})
%%timeit -r1 -n1
rfm = df.groupby(["master_id"]).agg({'last_order_date': lambda x: (today_date - x).dt.days,
                                        "frequency":  lambda x: x,
                                        "monetary":  lambda x: x})


rfm.columns = ['recency', 'frequency', 'monetary']

df["recency"] = (today_date - df["last_order_date"]).dt.days
rfm = df[['recency', 'frequency', 'monetary']]
rfm.index = df["master_id"]
rfm.describe().T

# GÖREV 3: RF ve RFM Skorlarının Hesaplanması

rfm["recency_score"] = pd.qcut(rfm["recency"],q=5, labels=[5,4,3,2,1])
rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"),q=5,  labels=[1,2,3,4,5])
rfm["monetary_score"] = pd.qcut(rfm["monetary"],q=5, labels=[1,2,3,4,5])

rfm[["frequency"]].describe([0,0.2,0.4,0.6,0.8]).T
rfm[["frequency"]].rank(method="first").describe([0,0.2,0.4,0.6,0.8]).T

rfm[["recency"]].describe([0,0.2,0.4,0.6,0.8]).T

pd.qcut(rfm["recency"],q=5).sort_values(ascending=True)

rfm["RF_SCORE"] = (rfm['recency_score'].astype(str) +
                    rfm['frequency_score'].astype(str))


# GÖREV 4: RF Skorlarının Segment Olarak Tanımlanması

seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm['segment'] = rfm['RF_SCORE'].replace(seg_map, regex=True)

#veya
rfm['segment'] = 0
for i,y in seg_map.items():
    rfm['segment'].loc[rfm['RF_SCORE'].str.contains(i)] = y

# GÖREV 5: Aksiyon zamanı!

# 1. Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz.
rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean"])

# 2. RFM analizi yardımı ile 2 case için ilgili profildeki müşterileri bulun ve müşteri id'lerini csv ye kaydediniz.
rfm[(rfm["segment"] == "cant_loose") |(rfm["segment"] == "champions")].head()


customers = pd.DataFrame()
customers["new_customer_id"] = rfm[(rfm["segment"] == "cant_loose") | (rfm["segment"] == "champions")].index

customers.to_csv("customers.csv",index=False)


# a. FLO bünyesine yeni bir kadın ayakkabı markası dahil ediyor. Dahil ettiği markanın ürün fiyatları genel müşteri tercihlerinin üstünde. Bu nedenle markanın
# tanıtımı ve ürün satışları için ilgilenecek profildeki müşterilerle özel olarak iletişime geçeilmek isteniliyor. Sadık müşterilerinden(champions,loyal_customers), ortalama 250 TL üzeri ve kadın kategorisinden alışveriş yapan kişiler özel olarak iletişim kuralacak müşteriler. Bu müşterilerin id numaralarını csv dosyasına yeni_marka_hedef_müşteri_id.cvs olarak kaydediniz.


pd.merge(rfm[(rfm["segment"] == "loyal_customers") |(rfm["segment"] == "champions")],df["master_id"][df["interested_in_categories_12"].str.contains("KADIN") ],how='inner', on="master_id")["master_id"].to_csv("yeni_marka_hedef_müşteri_id.csv", index=False)

r = rfm[rfm["segment"].isin(["loyal_customers","champions"])]
r = rfm[(rfm["segment"] == "loyal_customers") | (rfm["segment"] == "champions")]
d = df["master_id"][df["interested_in_categories_12"].str.contains("KADIN") & (df["monetary"]>250)]
pd.merge(r,d,how='inner', on="master_id")["master_id"].to_csv("customers.csv",index=False)


# b. Erkek ve Çoçuk ürünlerinde %40'a yakın indirim planlanmaktadır. Bu indirimle ilgili kategorilerle ilgilenen geçmişte iyi müşteri olan ama uzun süredir
# alışveriş yapmayan kaybedilmemesi gereken müşteriler, uykuda olanlar ve yeni gelen müşteriler özel olarak hedef alınmak isteniliyor. Uygun profildeki müşterilerin id'lerini csv dosyasına indirim_hedef_müşteri_ids.csv
# olarak kaydediniz.
pd.merge(rfm[(rfm["segment"] == "cant_loose") |(rfm["segment"] == "about_to_sleep") | (rfm["segment"] == "new_customers")],df[df["interested_in_categories_12"].str.contains(r".*(ERKEK.*COCUK|COCUK.*ERKEK)")],how='inner', on="master_id")["master_id"].to_csv("indirim_hedef_müşteri_ids.csv",index=False)

r = rfm[rfm["segment"].isin(["cant_loose","about_to_sleep","new_customers"])]
r = rfm[(rfm["segment"] == "cant_loose") |(rfm["segment"] == "about_to_sleep") | (rfm["segment"] == "new_customers")]

d= df[df["interested_in_categories_12"].str.contains(r".*(ERKEK.*COCUK|COCUK.*ERKEK)")]
d = df[df["interested_in_categories_12"].str.contains(r".*(ERKEK|COCUK)")]
#veya
d = df[df["interested_in_categories_12"].str.contains("COCUK") & df["interested_in_categories_12"].str.contains("ERKEK")]
d = df[df["interested_in_categories_12"].str.contains("COCUK") | df["interested_in_categories_12"].str.contains("ERKEK")]

pd.merge(r,d,how='inner', on="master_id")["master_id"].to_csv("indirim_hedef_müşteri_ids.csv",index=False)
#veya
d[d["master_id"].isin(r.index)]["master_id"]

# GÖREV 6: Tüm süreci fonksiyonlaştırınız.

def rfm_all(df, csv=False, return_rfm=False):
    df = data_prep(df)
    today_date = df["first_order_date"].max() + pd.Timedelta(days=2)
    df["recency"] = [i.days for i in (today_date - df["last_order_date"])]
    rfm = df[['recency', 'frequency', 'monetary']]
    rfm.index = df["master_id"]
    rfm["recency_score"] = pd.qcut(rfm["recency"], q=5, labels=[5, 4, 3, 2, 1])
    rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5])
    rfm["monetary_score"] = pd.qcut(rfm["monetary"], q=5, labels=[1, 2, 3, 4, 5])

    rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) +
                        rfm['frequency_score'].astype(str))
    seg_map = {
        r'[1-2][1-2]': 'hibernating',
        r'[1-2][3-4]': 'at_Risk',
        r'[1-2]5': 'cant_loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalists',
        r'5[4-5]': 'champions'
    }

    rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)

    customers = pd.DataFrame()
    customers["new_customer_id"] = rfm[(rfm["segment"] == "cant_loose") | (rfm["segment"] == "champions")].index
    yeni_marka_hedef_müşteri_id = pd.merge(rfm[(rfm["segment"] == "loyal_customers") | (rfm["segment"] == "champions")],
             df["master_id"][df["interested_in_categories_12"].str.contains("KADIN") & (df["monetary"] > 250)],
             how='inner', on="master_id")["master_id"]
    indirim_hedef_müşteri_ids = pd.merge(rfm[(rfm["segment"] == "cant_loose") | (rfm["segment"] == "about_to_sleep") | (
                rfm["segment"] == "new_customers")], df["master_id"][
                 df["interested_in_categories_12"].str.contains(r".*ERKEK.*COCUK.*")],
             how='inner', on="master_id")["master_id"]
    rfm_done = rfm[["recency", "frequency", "monetary", "segment"]]
    if csv:
        customers.to_csv("customers.csv", index=False)
        yeni_marka_hedef_müşteri_id.to_csv("yeni_marka_hedef_müşteri_id.csv", index=False)
        indirim_hedef_müşteri_ids.to_csv("indirim_hedef_müşteri_ids.csv", index=False)
    if return_rfm:
        return rfm_done

rfm_all(df, csv=True, return_rfm=False)
