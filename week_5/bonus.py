#QUE
# Association RuleBased Recommender System
#Aşağıda 3 farklı kullanıcının sepet bilgileri verilmiştir.
# Bu sepet bilgilerine en uygun ürün önerisini birliktelik kuralı kullanarak yapınız.
# Ürün önerileri 1 tane ya da 1'den fazla olabilir.
# Karar kurallarını 2010-2011 Germany müşterileri üzerinden türetiniz

#TODO
#    Kullanıcı 1’in sepetinde bulunan ürünün id'si: 21987
#    Kullanıcı 2’in sepetinde bulunan ürünün id'si : 23235
#    Kullanıcı 3’in sepetinde bulunan ürünün id'si : 22747
#QUE
# Veri Seti Hikayesi
#Online Retail II isimli veri seti İngiltere merkezli bir perakende şirketinin 01/12/2009 - 09/12/2011 tarihleri arasındaki online satış işlemlerini içeriyor.
# Şirketin ürün kataloğunda hediyelik eşyalar yer almaktadır ve çoğu müşterisinin toptancı olduğu bilgisi mevcuttur.

# InvoiceNo Fatura Numarası ( Eğer bu kod C ile başlıyorsa işlemin iptal edildiğini ifade eder )
# StockCode Ürün kodu ( Her bir ürün için eşsiz )
# Description Ürün ismi
# Quantity Ürün adedi ( Faturalardaki ürünlerden kaçar tane satıldığı)
# InvoiceDate Fatura tarihi
# UnitPrice Fatura fiyatı ( Sterlin )
# CustomerID Eşsiz müşteri numarası
# Country Ülke ismi

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
pd.options.display.max_columns=None
pd.options.display.max_rows=50
pd.options.display.float_format = '{:.3f}'.format
pd.options.display.width = 1000


#TODO
# Görev 1: Veriyi Hazırlama

# Adım 1: Online Retail II veri setinden 2010-2011 sheet’ini okutunuz.
pd.ExcelFile("week_5/bonus/online_retail_II.xlsx").sheet_names

df_ = pd.read_excel("week_5/bonus/online_retail_II.xlsx", sheet_name="Year 2010-2011")

df = df_.copy()

df = df[df["Country"]=="Germany"]
# Adım 2: StockCode’u POST olan gözlem birimlerini drop ediniz. (POST her faturaya eklenen bedel, ürünü ifade etmemektedir.)

df = df[~df["StockCode"].str.contains("POST", na=False)]

# Adım 3: Boş değer içeren gözlem birimlerini drop ediniz.

df.isnull().sum()
df.dropna(inplace = True)

# Adım 4: Invoice içerisinde C bulunan değerleri veri setinden çıkarınız. (C faturanın iptalini ifade etmektedir.)
df = df[~df["Invoice"].astype(str).str.contains("C")]

# Adım 5: Price değeri sıfırdan küçük olan gözlem birimlerini filtreleyiniz.
df.describe().T
df = df[df["Price"] > 0]

# Adım 6: Price ve Quantity değişkenlerinin aykırı değerlerini inceleyiniz, gerekirse baskılayınız.
df.describe([0.01,0.99]).T

def outliner(data,columns):
    data = data.copy()
    for col in columns:
        q1 = data[col].quantile(0.01)
        q3 = data[col].quantile(0.95)
        IQR = q3 - q1
        up = q3 + 1.5 * IQR
        low = q1 - 1.5 * IQR
        data.loc[data[col] > up, col] = up
        data.loc[data[col] < low, col] = low
    return data

df = outliner(df,columns=["Price","Quantity"])

#TODO
# Görev 2: Alman Müşteriler Üzerinden Birliktelik Kuralları Üretme

#Adım 1: Aşağıdaki gibi fatura ürün pivot table’i oluşturacak create_invoice_product_df fonksiyonunu tanımlayınız. (tablodaki fransız

create_invoice_product_df = df.pivot_table(index="Invoice",columns="StockCode",values="Customer ID", aggfunc="count").notnull()

create_invoice_product_df = df.groupby(['Invoice', "StockCode"])['Customer ID'].count().unstack().notnull()

#SO BUNU YAPMA
create_invoice_product_df = df.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)
# |:-----------:|:-----------------------:|:------------------------------:|:-----------------------:|
# | Description | NINE DRAWER OFFICE TIDY | SET 2 TEA TOWELS I LOVE LONDON | SPACEBOY BABY GIFT SET… |
# |:-----------:|:-----------------------:|:------------------------------:|:-----------------------:|
# |   Invoice   |                         |                                |                         |
# |    536370   |            0            |                1               |            0            |
# |    536852   |            1            |                0               |            1            |
# |    536974   |            0            |                0               |            0            |
# |    537065   |            1            |                0               |            0            |
# |    537463   |            0            |                0               |            1            |
# |:-----------:|:-----------------------:|:------------------------------:|:-----------------------:|
#Adım 2: Kuralları oluşturacak create_rules fonksiyonunu tanımlayınız ve alman müşteriler için kurallarını bulunuz.
def create_rules(data):
    data = data.copy()
    frequent_itemsets = apriori(data,min_support=0.01,use_colnames=True)
    rules = association_rules(frequent_itemsets,metric="support",min_threshold=0.01)
    return rules
rules = create_rules(create_invoice_product_df)

# Adım 1: check_id fonksiyonunu kullanarak verilen ürünlerin isimlerini bulunuz.

def check_id(data,id):
    name = data["Description"][data["StockCode"] == id].iloc[0]
    return name

check_id(df,id=21987)
check_id(df,id=23235)
check_id(df,id=22747)
# Adım 2: arl_recommender fonksiyonunu kullanarak 3 kullanıcı için ürün önerisinde bulununuz.

def arl_recommender1(rules_df, product_id, rec=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    conseq = sorted_rules["consequents"][sorted_rules["antecedents"].astype(str).str.contains(r"\b"+str(product_id),regex=True)]
    commons = list(dict.fromkeys([x for i in conseq for x in i]))[:rec]
    return commons

def arl_recommender2(rules_df, product_id, rec=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    explode_frozen = sorted_rules.explode("antecedents").explode("consequents")[["antecedents","consequents"]].drop_duplicates()
    commons = explode_frozen["consequents"][explode_frozen["antecedents"]==product_id].head(rec)
    return commons.values.flatten()

def arl_recommender(rules_df, product_id, rec=1):
    #dersteki
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec]
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


suggest_list = arl_recommender(rules, product_id=23235, rec=10)
suggest_list = arl_recommender1(rules, product_id=23235, rec=10)
# Adım 3: Önerilecek ürünlerin isimlerine bakınız.
check_id(df,id=23235)

for suggest in suggest_list:
    print(check_id(df,id=suggest))

