#############################################
# Kural Tabanlı Sınıflandırma ile Potansiyel Müşteri Getirisi Hesaplama
#############################################

#############################################
# İş Problemi
#############################################
# Bir oyun şirketi müşterilerinin bazı özelliklerini kullanarak seviye tabanlı (level based) yeni müşteri tanımları (persona)
# oluşturmak ve bu yeni müşteri tanımlarına göre segmentler oluşturup bu segmentlere göre yeni gelebilecek müşterilerin şirkete
# ortalama ne kadar kazandırabileceğini tahmin etmek istemektedir.

# Örneğin: Türkiye’den IOS kullanıcısı olan 25 yaşındaki bir erkek kullanıcının ortalama ne kadar kazandırabileceği belirlenmek isteniyor.


#############################################
# Veri Seti Hikayesi
#############################################
# Persona.csv veri seti uluslararası bir oyun şirketinin sattığı ürünlerin fiyatlarını ve bu ürünleri satın alan kullanıcıların bazı
# demografik bilgilerini barındırmaktadır. Veri seti her satış işleminde oluşan kayıtlardan meydana gelmektedir. Bunun anlamı tablo
# tekilleştirilmemiştir. Diğer bir ifade ile belirli demografik özelliklere sahip bir kullanıcı birden fazla alışveriş yapmış olabilir.

# Price: Müşterinin harcama tutarı
# Source: Müşterinin bağlandığı cihaz türü
# Sex: Müşterinin cinsiyeti
# Country: Müşterinin ülkesi
# Age: Müşterinin yaşı

################# Uygulama Öncesi #####################

#    PRICE   SOURCE   SEX COUNTRY  AGE
# 0     39  android  male     bra   17
# 1     39  android  male     bra   17
# 2     49  android  male     bra   17
# 3     29  android  male     tur   17
# 4     49  android  male     tur   17

################# Uygulama Sonrası #####################

#       customers_level_based        PRICE SEGMENT
# 0   BRA_ANDROID_FEMALE_0_18  1139.800000       A
# 1  BRA_ANDROID_FEMALE_19_23  1070.600000       A
# 2  BRA_ANDROID_FEMALE_24_30   508.142857       A
# 3  BRA_ANDROID_FEMALE_31_40   233.166667       C
# 4  BRA_ANDROID_FEMALE_41_66   236.666667       C


#############################################
# PROJE GÖREVLERİ
#############################################
import pandas as pd
import numpy as np
#############################################
# GÖREV 1: Aşağıdaki soruları yanıtlayınız.
#############################################

# Soru 1: persona.csv dosyasını okutunuz ve veri seti ile ilgili genel bilgileri gösteriniz.

df = pd.read_csv("week_2/persona.csv")

df.head()

df.info()

df.describe(include=[object]).T

df.isnull().sum()
# Soru 2: Kaç unique SOURCE vardır? Frekansları nedir?

df["SOURCE"].value_counts()
df["SOURCE"].mode()

# Soru 3: Kaç unique PRICE vardır?

df["PRICE"].nunique()


# Soru 4: Hangi PRICE'dan kaçar tane satış gerçekleşmiş?

df["PRICE"].value_counts()

# Soru 5: Hangi ülkeden kaçar tane satış olmuş?

df["COUNTRY"].value_counts()

# Soru 6: Ülkelere göre satışlardan toplam ne kadar kazanılmış?

df.groupby("COUNTRY").agg({"PRICE": "sum"})

df.groupby("COUNTRY")["PRICE"].sum()

df.groupby("COUNTRY")[["PRICE"]].sum()

df.groupby("COUNTRY")["PRICE"].agg([("Toplam_Kazanç","sum")])

df.pivot_table("PRICE","COUNTRY",aggfunc="sum")

# Soru 7: SOURCE türlerine göre satış sayıları nedir?

df.groupby("SOURCE")["SOURCE"].agg([("Satış_Sayısı","count")])

df.groupby("SOURCE").agg({"SOURCE": "count"})

df.groupby("SOURCE")["SOURCE"].count()

df["SOURCE"].value_counts()

df.pivot_table("PRICE","SOURCE",aggfunc="count")

# Soru 8: Ülkelere göre PRICE ortalamaları nedir?

df.groupby("COUNTRY")["PRICE"].agg([("Fiyat_Ortalamaları","mean")])

df.groupby("COUNTRY").agg({"PRICE": "mean"})

df.groupby("COUNTRY")["PRICE"].mean()

df.pivot_table("PRICE","COUNTRY",aggfunc="mean")

# Soru 9: SOURCE'lara göre PRICE ortalamaları nedir?

df.groupby("SOURCE")["PRICE"].agg([("Fiyat_Ortalamaları","mean")])

df.groupby("SOURCE").agg({"PRICE": "mean"})

df.groupby("SOURCE")["PRICE"].mean()

df.pivot_table("PRICE","SOURCE",aggfunc="mean")

# Soru 10: COUNTRY-SOURCE kırılımında PRICE ortalamaları nedir?

df.groupby(["COUNTRY","SOURCE"])["PRICE"].agg([("Fiyat_Ortalamaları","mean")])

df.groupby(["COUNTRY","SOURCE"]).agg({"PRICE": "mean"})

df.groupby(["COUNTRY","SOURCE"])["PRICE"].mean()

df.pivot_table("PRICE",["COUNTRY","SOURCE"],aggfunc="mean")

#############################################
# GÖREV 2: COUNTRY, SOURCE, SEX, AGE kırılımında ortalama kazançlar nedir?
#############################################
df.groupby(["COUNTRY","SOURCE","SEX","AGE"])["PRICE"].agg([("Kazanç_Ortalamaları","mean")]).head()

df.groupby(["COUNTRY","SOURCE","SEX","AGE"]).agg({"PRICE": "mean"})

df.groupby(["COUNTRY","SOURCE","SEX","AGE"])[["PRICE"]].mean().head()

df.groupby(["COUNTRY","SOURCE","SEX","AGE"])["PRICE"].mean().head()

df.pivot_table("PRICE",["COUNTRY","SOURCE","SEX","AGE"],aggfunc="mean").head()

[df.groupby(["COUNTRY","SOURCE","SEX","AGE"])[["PRICE"]].mean().apply(lambda x : x.loc[[i]].head(5)) for i in set(df.COUNTRY)]

#############################################
# GÖREV 3: Çıktıyı PRICE'a göre sıralayınız.
#############################################
# Önceki sorudaki çıktıyı daha iyi görebilmek için sort_values metodunu azalan olacak şekilde PRICE'a uygulayınız.
# Çıktıyı agg_df olarak kaydediniz.

agg_df = df.groupby(["COUNTRY","SOURCE","SEX","AGE"]).agg({"PRICE": "mean"}).sort_values(by="PRICE",ascending=False)


#############################################
# GÖREV 4: Indekste yer alan isimleri değişken ismine çeviriniz.
#############################################
# Üçüncü sorunun çıktısında yer alan PRICE dışındaki tüm değişkenler index isimleridir.
# Bu isimleri değişken isimlerine çeviriniz.
# İpucu: reset_index()
# agg_df.reset_index(inplace=True)

agg_df.head()

agg_df.reset_index(inplace=True)

#############################################
# GÖREV 5: AGE değişkenini kategorik değişkene çeviriniz ve agg_df'e ekleyiniz.
#############################################
# Age sayısal değişkenini kategorik değişkene çeviriniz.
# Aralıkları ikna edici olacağını düşündüğünüz şekilde oluşturunuz.
# Örneğin: '0_18', '19_23', '24_30', '31_40', '41_70'


agg_df["AGE_CAT"] = pd.cut(agg_df["AGE"],bins=[0,18,23,30,40,agg_df["AGE"].max()],
                                        labels=['0_18', '19_23', '24_30', '31_40', f'41_{agg_df["AGE"].max()}'])

agg_df["AGE_CAT"].value_counts()
#############################################
# GÖREV 6: Yeni level based müşterileri tanımlayınız ve veri setine değişken olarak ekleyiniz.
#############################################
# customers_level_based adında bir değişken tanımlayınız ve veri setine bu değişkeni ekleyiniz.
# Dikkat!
# list comp ile customers_level_based değerleri oluşturulduktan sonra bu değerlerin tekilleştirilmesi gerekmektedir.
# Örneğin birden fazla şu ifadeden olabilir: USA_ANDROID_MALE_0_18
# Bunları groupby'a alıp price ortalamalarını almak gerekmektedir.


agg_df["customers_level_based"] = [f"{COUNTRY}_{SOURCE}_{SEX}_{AGE_CAT}".upper()
                                 for COUNTRY,SOURCE,SEX,AGE_CAT
                                 in zip(agg_df["COUNTRY"],agg_df["SOURCE"],agg_df["SEX"],agg_df["AGE_CAT"])]

agg_df["customers_level_based"] = [f"{i[0]}_{i[1]}_{i[2]}_{i[3]}".upper()
                                   for _,i
                                   in agg_df[["COUNTRY","SOURCE","SEX","AGE_CAT"]].iterrows()]

agg_df["customers_level_based"] = np.vectorize(lambda COUNTRY,SOURCE,SEX,AGE_CAT:
                                    f"{COUNTRY}_{SOURCE}_{SEX}_{AGE_CAT}".upper()) \
                                    (agg_df["COUNTRY"],agg_df["SOURCE"],agg_df["SEX"],agg_df["AGE_CAT"])
#Akif
agg_df['customers_level_based'] = agg_df[['COUNTRY', 'SOURCE', 'SEX', 'age_cat']].agg(lambda x: '_'.join(x).upper(), axis=1)

#Ahmet
agg_df["customers_level_based"] = [f"{i[0]}_{i[1]}_{i[2]}_{i[5]}".upper() for i in agg_df.values]


agg_df["customers_level_based"] = (agg_df["COUNTRY"]+"_"+agg_df["SOURCE"]+"_"+agg_df["SEX"]+"_"+agg_df["AGE_CAT"].astype("O")).str.upper()



agg_df = agg_df.groupby("customers_level_based").agg({"PRICE":"mean"})
#############################################
# GÖREV 7: Yeni müşterileri (USA_ANDROID_MALE_0_18) segmentlere ayırınız.
#############################################
# PRICE'a göre segmentlere ayırınız,
# segmentleri "SEGMENT" isimlendirmesi ile agg_df'e ekleyiniz,
# segmentleri betimleyiniz,

agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"],q=4,labels=["D","C","B","A"])

#############################################
# GÖREV 8: Yeni gelen müşterileri sınıflandırınız ne kadar gelir getirebileceğini tahmin ediniz.
#############################################
# 33 yaşında ANDROID kullanan bir Türk kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?
#agg_df.reset_index(inplace = True)
resetli_df=agg_df.reset_index()
new_customer = "TUR_ANDROID_FEMALE_31_40"

agg_df.head()
agg_df.loc[new_customer]

resetli_df.head()
resetli_df[resetli_df["customers_level_based"]==new_customer][["PRICE","SEGMENT"]]

# 35 yaşında IOS kullanan bir Fransız kadını hangi segmente ve ortalama ne kadar gelir kazandırması beklenir?

neu_customer = "FRA_IOS_FEMALE_31_40"

agg_df.loc[neu_customer]

resetli_df[["PRICE","SEGMENT"]][resetli_df["customers_level_based"]==neu_customer]


# All in one func
df = pd.read_csv("week_2\persona.csv")

def all_in_one(df , customer):
    df = df.groupby(["COUNTRY","SOURCE","SEX","AGE"]).agg({"PRICE": "mean"}).sort_values(by="PRICE",ascending=False).reset_index()
    df["AGE_CAT"] = pd.cut(df["AGE"], bins=[0, 18, 23, 30, 40, df["AGE"].max()],
                                        labels=['0_18', '19_23', '24_30', '31_40', f'41_{df["AGE"].max()}'])
    df["customers_level_based"] = [f"{i[0]}_{i[1]}_{i[2]}_{i[5]}".upper() for i in df.values]
    df = df.groupby("customers_level_based").agg({"PRICE": "mean"})
    df["SEGMENT"] = pd.qcut(df["PRICE"], q=4, labels=["D", "C", "B", "A"])
    df.reset_index(inplace=True)
    result = df[df["customers_level_based"] == customer][["PRICE", "SEGMENT"]]
    return (f"{new_customer} Ortalama Gelir: {result.iloc[0][0].round(2)}, Segment: {result.iloc[0][1]}")


new_customer = "TUR_ANDROID_FEMALE_31_40"
new_customer = "FRA_IOS_FEMALE_31_40"

all_in_one(df,new_customer)


