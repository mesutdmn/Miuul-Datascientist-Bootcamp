
#########################
# İş Problemi
#########################

# Türkiye’nin en büyük online hizmet platformu olan Armut, hizmet verenler ile hizmet almak isteyenleri buluşturmaktadır.
# Bilgisayarın veya akıllı telefonunun üzerinden birkaç dokunuşla temizlik, tadilat, nakliyat gibi hizmetlere kolayca
# ulaşılmasını sağlamaktadır.
# Hizmet alan kullanıcıları ve bu kullanıcıların almış oldukları servis ve kategorileri içeren veri setini kullanarak
# Association Rule Learning ile ürün tavsiye sistemi oluşturulmak istenmektedir.


#########################
# Veri Seti
#########################
#Veri seti müşterilerin aldıkları servislerden ve bu servislerin kategorilerinden oluşmaktadır.
# Alınan her hizmetin tarih ve saat bilgisini içermektedir.

# UserId: Müşteri numarası
# ServiceId: Her kategoriye ait anonimleştirilmiş servislerdir. (Örnek : Temizlik kategorisi altında koltuk yıkama servisi)
# Bir ServiceId farklı kategoriler altında bulanabilir ve farklı kategoriler altında farklı servisleri ifade eder.
# (Örnek: CategoryId’si 7 ServiceId’si 4 olan hizmet petek temizliği iken CategoryId’si 2 ServiceId’si 4 olan hizmet mobilya montaj)
# CategoryId: Anonimleştirilmiş kategorilerdir. (Örnek : Temizlik, nakliyat, tadilat kategorisi)
# CreateDate: Hizmetin satın alındığı tarih




#########################
# GÖREV 1: Veriyi Hazırlama
#########################
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


import numpy as np
import timeit


pd.options.display.max_columns=None
pd.options.display.max_rows=10
pd.options.display.float_format = '{:.3f}'.format
pd.options.display.width = 1000

# Adım 1: armut_data.csv dosyasınız okutunuz.
df = pd.read_csv("week_5/armut_data.csv")
df.shape
df.head()

df["UserId"].nunique()
# Adım 2: ServisID her bir CategoryID özelinde farklı bir hizmeti temsil etmektedir.

# ServiceID ve CategoryID'yi "_" ile birleştirerek hizmetleri temsil edecek yeni bir değişken oluşturunuz.

%timeit -r1 -n3 df["Hizmet"] = [f"{O}_{o}" for O,o in zip(df["ServiceId"],df["CategoryId"])]

%timeit -r1 -n3 df["Hizmet"] = np.vectorize(lambda O,o : f"{O}_{o}")(df["ServiceId"],df["CategoryId"])

%timeit -r1 -n3 df["Hizmet"] = df["ServiceId"].astype(str)+"_"+df["CategoryId"].astype(str)

%timeit -r1 -n3 df["Hizmet"] = [f"{i[0]}_{i[1]}" for i in df[['ServiceId', 'CategoryId']].astype(str).values] #Ahmet

%timeit -r1 -n3 df["Hizmet"] = df[['ServiceId', 'CategoryId']].astype(str).apply(lambda x: '_'.join(x), axis=1)

%timeit -r1 -n3 df["Hizmet"] = df[['ServiceId', 'CategoryId']].astype(str).agg(lambda x: '_'.join(x), axis=1) #Akif hoca

# Adım 3: Veri seti hizmetlerin alındığı tarih ve saatten oluşmaktadır, herhangi bir sepet tanımı (fatura vb. ) bulunmamaktadır.
# Association Rule Learning uygulayabilmek için bir sepet (fatura vb.) tanımı oluşturulması gerekmektedir.
# Burada sepet tanımı her bir müşterinin aylık aldığı hizmetlerdir. Örneğin; 7256 id'li müşteri 2017'in 8.ayında aldığı 9_4, 46_4 hizmetleri bir sepeti;
# 2017’in 10.ayında aldığı  9_4, 38_4  hizmetleri başka bir sepeti ifade etmektedir. Sepetleri unique bir ID ile tanımlanması gerekmektedir.
# Bunun için öncelikle sadece yıl ve ay içeren yeni bir date değişkeni oluşturunuz. UserID ve yeni oluşturduğunuz date değişkenini "_"
# ile birleştirirek ID adında yeni bir değişkene atayınız.

df["CreateDate"] = pd.to_datetime(df["CreateDate"])

df["new_date"] = df['CreateDate'].dt.to_period('M')

df["SepetID"] = [f"{O}_{o}" for O,o in zip(df["UserId"],df["new_date"])]

#########################
# GÖREV 2: Birliktelik Kuralları Üretiniz
#########################

# Adım 1: Aşağıdaki gibi sepet hizmet pivot table’i oluşturunuz.

# Hizmet         0_8  10_9  11_11  12_7  13_11  14_7  15_1  16_8  17_5  18_4..
# SepetID
# 0_2017-08        0     0      0     0      0     0     0     0     0     0..
# 0_2017-09        0     0      0     0      0     0     0     0     0     0..
# 0_2018-01        0     0      0     0      0     0     0     0     0     0..
# 0_2018-04        0     0      0     0      0     1     0     0     0     0..
# 10000_2017-08    0     0      0     0      0     0     0     0     0     0..

df.info()

#eski, hiçbirini seçme :D
pivot = df.pivot_table(index="SepetID",columns="Hizmet",values="UserId", aggfunc="count", fill_value=0).applymap(lambda x: True if x>0 else False).astype(int)
pivoteddf = df.groupby(["SepetID","Hizmet"])["Hizmet"].count().unstack().fillna(0).applymap(lambda x: 1 if x>0 else 0)

#yeni, birini seç
pivot = df.pivot_table(index="SepetID",columns="Hizmet",values="UserId", aggfunc="count").notnull()
pivoteddf2 = df.groupby(["SepetID","Hizmet"])["Hizmet"].count().unstack().notnull()

# Adım 2: Birliktelik kurallarını oluşturunuz.

frequent_itemsets = apriori(pivot,min_support=0.01,use_colnames=True)
frequent_itemsets.sort_values("support", ascending=False)


rules = association_rules(frequent_itemsets,metric="support",min_threshold=0.01)

#Adım 3: arl_recommender fonksiyonunu kullanarak en son 2_0 hizmetini alan bir kullanıcıya hizmet önerisinde bulununuz.

def arl_recommender(rules_df, product_id, rec_count=1):
    #dersteki
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]
def arl_recommender(rules_df, product_id, rec=1):
    #olması gereken
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id :
                for k in list(sorted_rules.iloc[i]["consequents"]):
                    if k not in recommendation_list:
                        recommendation_list.append(k)

    return recommendation_list[0:rec]

def arl_recommender1(rules_df, product_id, rec=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    conseq = sorted_rules["consequents"][sorted_rules["antecedents"].astype(str).str.contains(r"\b"+str(product_id),regex=True)]
    commons = list(dict.fromkeys([x for i in conseq for x in i]))[:rec]
    return commons

arl_recommender1(rules, "2_0", 3)

def arl_recommender2(rules_df, product_id, rec=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    explode_frozen = sorted_rules.explode("antecedents").explode("consequents")[["antecedents","consequents"]].drop_duplicates()
    commons = explode_frozen["consequents"][explode_frozen["antecedents"]==product_id].head(rec)
    return commons.values.flatten().tolist()

arl_recommender2(rules, "2_0", 3)


arl_recommender(rules, "2_0", 5)
arl_recommender1(rules, "2_0", 5)
arl_recommender2(rules, "2_0", 5)