

##################################################
# Pandas Alıştırmalar
##################################################

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

#########################################
# Görev 1: Seaborn kütüphanesi içerisinden Titanic veri setini tanımlayınız.
#########################################

df = sns.load_dataset("titanic")
df.head()
#########################################
# Görev 2: Yukarıda tanımlanan Titanic veri setindeki kadın ve erkek yolcuların sayısını bulunuz.
#########################################

df["sex"].value_counts()

#########################################
# Görev 3: Her bir sutuna ait unique değerlerin sayısını bulunuz.
#########################################

pd.DataFrame(df.nunique()).T

#########################################
# Görev 4: pclass değişkeninin unique değerleri bulunuz.
#########################################

df["pclass"].unique()


#########################################
# Görev 5:  pclass ve parch değişkenlerinin unique değerlerinin sayısını bulunuz.
#########################################

df[["pclass","parch"]].nunique()

df[["pclass","parch"]].value_counts()

df["pclass"].value_counts()
df["parch"].value_counts()

#########################################
# Görev 6: embarked değişkeninin tipini kontrol ediniz. Tipini category olarak değiştiriniz. Tekrar tipini kontrol ediniz.
#########################################
df["embarked"].dtype

df["embarked"] = df["embarked"].astype("category")

df["embarked"].dtype
#########################################
# Görev 7: embarked değeri C olanların tüm bilgelerini gösteriniz.
#########################################

df[df["embarked"]=="C"].head()


#########################################
# Görev 8: embarked değeri S olmayanların tüm bilgelerini gösteriniz.
#########################################

df[df["embarked"]!="S"].head()

#########################################
# Görev 9: Yaşı 30 dan küçük ve kadın olan yolcuların tüm bilgilerini gösteriniz.
#########################################

df[(df["age"]<30) & (df["sex"]=="female")].head()


#########################################
# Görev 10: Fare'i 500'den büyük veya yaşı 70 den büyük yolcuların bilgilerini gösteriniz.
#########################################

df[(df["age"]>70) | (df["fare"]>500)].head()

#########################################
# Görev 11: Her bir değişkendeki boş değerlerin toplamını bulunuz.
#########################################

pd.DataFrame(df.isnull().sum()).T


#########################################
# Görev 12: who değişkenini dataframe'den düşürün.
#########################################

df.drop("who", axis=1, inplace=True)

df.head(5)
#########################################
# Görev 13: deck değikenindeki boş değerleri deck değişkenin en çok tekrar eden değeri (mode) ile doldurunuz.
#########################################
df["deck"].head()

type(df["deck"].mode())
df["deck"].fillna(mod, inplace=True)
df.head()
df.fillna(df[["deck"]].mode().loc[0])
type(df[["deck"]].mode())
#########################################
# Görev 14: age değikenindeki boş değerleri age değişkenin medyanı ile doldurun.
#########################################
df["age"].head(6)
type(df["age"].median())
df["age"].fillna(df["age"].median(), inplace=True)
df["age"] = df["age"].fillna(df["age"].median())
df = df.fillna(df[["age"]].median())
#########################################
# Görev 15: survived değişkeninin Pclass ve Cinsiyet değişkenleri kırılımınında sum, count, mean değerlerini bulunuz.
#########################################

df.groupby(["pclass","sex"]).agg({"survived":["sum","count","mean"]})

#########################################
# Görev 16:  30 yaşın altında olanlar 1, 30'a eşit ve üstünde olanlara 0 vericek bir fonksiyon yazınız.
# Yazdığınız fonksiyonu kullanarak titanik veri setinde age_flag adında bir değişken oluşturunuz oluşturunuz.
# (apply ve lambda yapılarını kullanınız)
#########################################

df["age_flag"]=df["age"].apply(lambda x: 1 if x<30 else 0)

df["age_flag"]=[1 if age<30 else 0 for age in df["age"]]

df["age_flag"]=np.vectorize(lambda x: 1 if x<30 else 0)(df['age'])

df["age_flag"].head()
#########################################
# Görev 17: Seaborn kütüphanesi içerisinden Tips veri setini tanımlayınız.
#########################################

tips = sns.load_dataset("tips")
tips.head()

#########################################
# Görev 18: Time değişkeninin kategorilerine (Dinner, Lunch) göre total_bill  değerlerinin toplamını, min, max ve ortalamasını bulunuz.
#########################################

tips.groupby("time").agg({"total_bill":["sum","min","max","mean"]})

#########################################
# Görev 19: Günlere ve time göre total_bill değerlerinin toplamını, min, max ve ortalamasını bulunuz.
#########################################

tips.groupby(["day","time"]).agg({"total_bill":["sum","min","max","mean"]})

#########################################
# Görev 20:Lunch zamanına ve kadın müşterilere ait total_bill ve tip  değerlerinin day'e göre toplamını, min, max ve ortalamasını bulunuz.
#########################################

tips[(tips["time"]=="Lunch") & (tips["sex"]=="Female")].groupby("day").agg({"total_bill" : ["sum","min","max","mean"],
                                                                            "tip": ["sum","min","max","mean"]})

#########################################
# Görev 21: size'i 3'ten küçük, total_bill'i 10'dan büyük olan siparişlerin ortalaması nedir?
#########################################

tips[(tips["size"]<3) & (tips["total_bill"]>10)].agg({"total_bill" : ["mean"]})

tips[(tips["size"]<3) & (tips["total_bill"]>10)]["total_bill"].mean()

#########################################
# Görev 22: total_bill_tip_sum adında yeni bir değişken oluşturun. Her bir müşterinin ödediği totalbill ve tip in toplamını versin.
#########################################

tips["total_bill_tip_sum"]=tips["total_bill"]+tips["tip"]

#########################################
# Görev 23: total_bill_tip_sum değişkenine göre büyükten küçüğe sıralayınız ve ilk 30 kişiyi yeni bir dataframe'e atayınız.
#########################################

new_df = tips.sort_values(by="total_bill_tip_sum", ascending=False, ignore_index=True).head(30)
new_df.head()
