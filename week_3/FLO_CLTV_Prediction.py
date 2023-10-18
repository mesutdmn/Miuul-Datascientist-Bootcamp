##############################################################
# BG-NBD ve Gamma-Gamma ile CLTV Prediction
##############################################################

###############################################################
# İş Problemi (Business Problem)
###############################################################
# FLO satış ve pazarlama faaliyetleri için roadmap belirlemek istemektedir.
# Şirketin orta uzun vadeli plan yapabilmesi için var olan müşterilerin gelecekte şirkete sağlayacakları potansiyel değerin tahmin edilmesi gerekmektedir.


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
# GÖREV 1: Veriyi Hazırlama
###############################################################
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
pd.options.display.max_columns=None
pd.options.display.float_format = '{:.3f}'.format
pd.options.display.width = 1000


# 1. flo_data_20K.csv verisini okuyunuz.Dataframe’in kopyasını oluşturunuz.
df_ = pd.read_csv("flo_data_20K.csv")

df = df_.copy()


# 2. Aykırı değerleri baskılamak için gerekli olan outlier_thresholds ve replace_with_thresholds fonksiyonlarını tanımlayınız.
# Not: cltv hesaplanırken frequency değerleri integer olması gerekmektedir.Bu nedenle alt ve üst limitlerini round() ile yuvarlayınız.
df.describe([0.01,0.99]).T
def outlier_thresholds(df, column):
    q1 = df[column].quantile(0.01)
    q3 = df[column].quantile(0.99)
    IQR = q3 - q1
    upper_limit = q3 + 1.5 * IQR
    lower_limit = q1 - 1.5 * IQR
    return upper_limit,lower_limit

def replace_with_thresholds(df, column):
    upper_limit, lower_limit = outlier_thresholds(df,column)
    df.loc[df[column]>upper_limit,column] = upper_limit
    df.loc[df[column]<lower_limit,column] = lower_limit

# 3. "order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online" değişkenlerinin
#aykırı değerleri varsa baskılayanız.
df.describe([0.01,0.99]).T

for i in df.select_dtypes(include=['number']).columns:
    replace_with_thresholds(df,i)


df.describe([0.01,0.99]).T
# 4. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir.
# Herbir müşterinin toplam alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.

df["frequency"] = round(df["order_num_total_ever_online"] + df["order_num_total_ever_offline"])
df["monetary"] = round(df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"])

# 5. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
df[df.columns[df.columns.str.contains("date")]] = df[df.columns[df.columns.str.contains("date")]].apply(pd.to_datetime)
#veya
for i in df.columns:
    if "date" in i:
        df[i] = pd.to_datetime(df[i])
###############################################################
# GÖREV 2: CLTV Veri Yapısının Oluşturulması
###############################################################

# 1.Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olarak alınız.
today_date = df["last_order_date"].max() + pd.Timedelta(days=2)

# 2.customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg değerlerinin yer aldığı yeni bir cltv dataframe'i oluşturunuz.

df["recency_cltv_weekly"] = (df["last_order_date"] - df["first_order_date"]).dt.days // 7
df["T_weekly"] = (today_date - df["first_order_date"]).dt.days // 7
cltv_df = df[["recency_cltv_weekly", "T_weekly", "frequency", "monetary"]]
cltv_df["monetary_cltv_avg"] = round(df["monetary"] / df["frequency"])
cltv_df.index = df["master_id"]

#veya
df["recency"] = (df["last_order_date"] - df["first_order_date"]).dt.days
cltv_grupped = df.groupby("master_id").agg(recency_cltv_weekly=("recency", lambda x : x // 7),
                            T_weekly=("first_order_date", lambda x : (today_date - x).dt.days // 7),
                            frequency=("frequency", lambda x: x),
                            monetary_cltv_avg=("monetary", lambda x:x)
                            )
cltv_grupped["monetary_cltv_avg"] = round(cltv_grupped["monetary_cltv_avg"] / cltv_grupped2["frequency"])

#veya
df["recency"] = (df["last_order_date"] - df["first_order_date"]).dt.days
cltv_grupped2 = df.groupby("master_id").agg({"recency": lambda x: x // 7,
                                             "first_order_date": lambda x: (today_date - x).dt.days // 7,
                                             "frequency": lambda x: x,
                                             "monetary": lambda x: x})

cltv_grupped2.columns = ["recency_cltv_weekly","T_weekly","frequency","monetary_cltv_avg"]
cltv_grupped2["monetary_cltv_avg"] = round(cltv_grupped2["monetary_cltv_avg"] / cltv_grupped2["frequency"])

cltv_df.describe().T

###############################################################
# GÖREV 3: BG/NBD, Gamma-Gamma Modellerinin Kurulması, 6 aylık CLTV'nin hesaplanması
###############################################################

# 1. BG/NBD modelini kurunuz.
bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['frequency'],
        cltv_df['recency_cltv_weekly'],
        cltv_df['T_weekly'])

# 3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_3_month olarak cltv dataframe'ine ekleyiniz.

cltv_df["exp_sales_3_month"] = bgf.predict(4 * 3,
                                            cltv_df['frequency'],
                                            cltv_df['recency_cltv_weekly'],
                                            cltv_df['T_weekly'])

# 6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_6_month olarak cltv dataframe'ine ekleyiniz.

cltv_df["exp_sales_6_month"] = bgf.predict(4 * 6,
                                            cltv_df['frequency'],
                                            cltv_df['recency_cltv_weekly'],
                                            cltv_df['T_weekly'])


# 3. ve 6.aydaki en çok satın alım gerçekleştirecek 10 kişiyi inceleyeniz.
cltv_df["exp_sales_3_month"].sort_values(ascending=False).head(10)
cltv_df["exp_sales_6_month"].sort_values(ascending=False).head(10)
plot_period_transactions(bgf)
plt.show(block=True)
# 2.  Gamma-Gamma modelini fit ediniz. Müşterilerin ortalama bırakacakları değeri tahminleyip exp_average_value olarak cltv dataframe'ine ekleyiniz.
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'],
        cltv_df['monetary'])

cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                        cltv_df['monetary'])

# 3. 6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz.

cltv_df["cltv"] = ggf.customer_lifetime_value(bgf,
                                               cltv_df['frequency'],
                                               cltv_df['recency_cltv_weekly'],
                                               cltv_df['T_weekly'],
                                               cltv_df['monetary'],
                                               time= 6,  # 6 aylık
                                               freq="W"
                                               )

# CLTV değeri en yüksek 20 kişiyi gözlemleyiniz.
cltv_df["cltv"].sort_values(ascending=False).head(20)


###############################################################
# GÖREV 4: CLTV'ye Göre Segmentlerin Oluşturulması
###############################################################

# 1. 6 aylık CLTV'ye göre tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz.
# cltv_segment ismi ile atayınız.

cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])

# 2. Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz.
#  4 grup içerisinden seçeceğiniz 2 grup için yönetime kısa kısa 6 aylık aksiyon önerilerinde bulununuz
cltv_df.groupby("cltv_segment").agg({"recency_cltv_weekly":["mean"],
                                     "frequency":["mean"],
                                     "monetary":["mean"]})

#6 aylık veriye göre segment A ve B deki müşterilerimizin satın alma ihtimali en yüksek müşteriler olduğu görülmektedir, kendilerine mail, mesaj gibi mediumlar aracılığı ile ulaşılıp kampyanlarımızdan veya ilgilendikleri alanlarla ilgili bilgilendirmede bulunursak satışlarımız artabilir.

# BONUS: Tüm süreci fonksiyonlaştırınız.
def all_in_one(df, bgf_month, cltv_month, csv=False, cltv_return=False):
    for i in df.select_dtypes(include=['number']).columns:
        replace_with_thresholds(df, i)
    df["frequency"] = (df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]).astype(int)
    df["monetary"] = (df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]).astype(int)
    df[df.columns[df.columns.str.contains("date")]] = df[df.columns[df.columns.str.contains("date")]].apply(
        pd.to_datetime)
    today_date = df["last_order_date"].max() + pd.Timedelta(days=2)
    df["recency_cltv_weekly"] = [i.days // 7 for i in (df["last_order_date"] - df["first_order_date"])]
    df["T_weekly"] = [i.days // 7 for i in (today_date - df["first_order_date"])]
    cltv_df = df[["recency_cltv_weekly", "T_weekly", "frequency", "monetary"]]
    cltv_df.index = df["master_id"]
    bgf = BetaGeoFitter(penalizer_coef=0.001)

    bgf.fit(cltv_df['frequency'],
            cltv_df['recency_cltv_weekly'],
            cltv_df['T_weekly'])
    if bgf_month:
        cltv_df[f"exp_sales_{bgf_month}_month"] = bgf.predict(4 * bgf_month,
                                                   cltv_df['frequency'],
                                                   cltv_df['recency_cltv_weekly'],
                                                   cltv_df['T_weekly'])
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(cltv_df['frequency'],
            cltv_df['monetary'])

    cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                           cltv_df['monetary'])
    if cltv_month:
        cltv_df["cltv"] = ggf.customer_lifetime_value(bgf,
                                                      cltv_df['frequency'],
                                                      cltv_df['recency_cltv_weekly'],
                                                      cltv_df['T_weekly'],
                                                      cltv_df['monetary'],
                                                      time=cltv_month,  # aylık
                                                      freq="W"
                                                      )
        cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])

    if csv:
        cltv_df.to_csv("cltv_df.csv")

    if cltv_return:
        return cltv_df


all_in_one(df.copy(), bgf_month = 3, cltv_month = 6, csv=True, cltv_return=True)






