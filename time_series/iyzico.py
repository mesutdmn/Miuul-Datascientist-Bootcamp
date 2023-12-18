###############################################################
# İş Problemi
###############################################################

# Iyzico internetten alışveriş deneyimini hem alıcılar hem de satıcılar için kolaylaştıran bir finansal teknolojiler şirketidir.
# E-ticaret firmaları, pazaryerleri ve bireysel kullanıcılar için ödeme altyapısı sağlamaktadır.
# 2020 yılının son 3 ayı için merchant_id ve gün bazında toplam işlem hacmi tahmini yapılması beklenmekte.


###############################################################
# Veri Seti Hikayesi
###############################################################
#  7 üye iş yerinin 2018’den 2020’e kadar olan verileri yer almaktadır.

# Transaction : İşlem sayısı
# MerchantID : Üye iş yerlerinin id'leri
# Paid Price : Ödeme miktarı

###############################################################
# GÖREVLER
###############################################################
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
pd.options.display.max_columns=100
pd.options.display.max_rows=20
pd.options.display.width=1000
pd.options.display.float_format = '${:,.2f}'.format
# Görev 1 : Veri Setinin Keşfi
# 1. iyzico_data.csv dosyasını okutunuz. transaction_date değişkeninin tipini date'e çeviriniz.
df = pd.read_csv("time_series/iyzico_data.csv").drop("Unnamed: 0", axis=1)
df["transaction_date"] = pd.to_datetime(df["transaction_date"])
df.head()
df.info()
# 2.Veri setinin başlangıc ve bitiş tarihleri nedir?
df["transaction_date"].min() #Timestamp('2018-01-01 00:00:00')
df["transaction_date"].max() #Timestamp('2020-12-31 00:00:00')
# 3.Her üye iş yerindeki toplam işlem sayısı kaçtır?
df.groupby("merchant_id")["Total_Transaction"].sum()

# 4.Her üye iş yerindeki toplam ödeme miktarı kaçtır?
df.groupby("merchant_id")["Total_Paid"].sum()

# 5.Her üye iş yerinin her bir yıl içerisindeki transaction count grafiklerini gözlemleyiniz.
df.groupby([df["transaction_date"].dt.year, df["merchant_id"]])["Total_Transaction"].sum().unstack().plot(kind="barh")
plt.show()
# Görev 2 : Feature Engineering tekniklerini uygulayanız. Yeni feature'lar türetiniz.
# Date Features
df["month"] = df["transaction_date"].dt.month
df["day_of_month"] = df["transaction_date"].dt.day
df["day_of_year"] = df["transaction_date"].dt.dayofyear
df["week_of_year"] = df["transaction_date"].dt.isocalendar().week
df["day_of_week"] = df["transaction_date"].dt.weekday
df["year"] = df["transaction_date"].dt.year
df["is_weekend"] = df["transaction_date"].dt.weekday > 4
df["is_month_start"] = df["transaction_date"].dt.is_month_start
df["is_month_end"] = df["transaction_date"].dt.is_month_end
df['quarter'] = df["transaction_date"].dt.quarter
# Lag/Shifted Features
def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe),))

def lag_features(dataframe, lags):
    for lag in lags:
        dataframe['sales_lag_' + str(lag)] = dataframe.groupby(["merchant_id"])['Total_Transaction'].transform(
            lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe

df = lag_features(df, [91,92,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,
                       350,351,352,352,354,355,356,357,358,359,360,361,362,363,364,365,366,367,368,369,370,
                       538,539,540,541,542,
                       718,719,720,721,722])
# Rolling Mean Features
def roll_mean_features(dataframe, windows):
    for window in windows:
        dataframe['sales_roll_mean_' + str(window)] = dataframe.groupby("merchant_id")['Total_Transaction']. \
                                                          transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise(
            dataframe)
    return dataframe

df = roll_mean_features(df, [91,92,178,179,180,181,182,359,360,361,449,450,451,539,540,541,629,630,631,720])


# Exponentially Weighted Mean Features
def ewm_features(dataframe, alphas, lags):
    for alpha in alphas:
        for lag in lags:
            dataframe['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby("merchant_id")['Total_Transaction'].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe

alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
lags = [91,92,178,179,180,181,182,359,360,361,449,450,451,539,540,541,629,630,631,720]

df = ewm_features(df, alphas, lags)
# Görev 3 : Modellemeye Hazırlık

# 1.One-hot encoding yapınız.
df = pd.get_dummies(df, columns=['merchant_id','day_of_week', 'month'])
df['Total_Transaction'] = np.log1p(df["Total_Transaction"].values)
# 2.Custom Cost Function'ları tanımlayınız.
def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200 * np.sum(num / denom)) / n
    return smape_val

def lgbm_smape(preds, train_data):
    labels = train_data.get_label()
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False
# 3.Veri setini train ve validation olarak ayırınız.
import re
df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

# 2020'nin 10.ayına kadar train seti.
train = df.loc[(df["transaction_date"] < "2020-10-01"), :]

# 2020'nin son 3 ayı validasyon seti.
val = df.loc[(df["transaction_date"] >= "2020-10-01"), :]

cols = [col for col in train.columns if col not in ['transaction_date', 'id', "Total_Transaction","Total_Paid", "year" ]]

Y_train = train['Total_Transaction']
X_train = train[cols]

Y_val = val['Total_Transaction']
X_val = val[cols]

# kontrol
Y_train.shape, X_train.shape, Y_val.shape, X_val.shape

# Görec 4 : LightGBM Modelini oluşturunuz ve SMAPE ile hata değerini gözlemleyiniz.
lgb_params = {'metric': {'mae'},
              'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': -1,
              'num_boost_round': 1000,
              'early_stopping_rounds': 200,
              'nthread': -1}
import lightgbm as lgb
lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)
lgbval = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=cols)

model = lgb.train(lgb_params, lgbtrain,
                  valid_sets=[lgbtrain, lgbval],
                  feval=lgbm_smape)

y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)

smape(np.expm1(y_pred_val), np.expm1(Y_val))

lgb.plot_importance(model, max_num_features=15)
plt.subplots_adjust(left=0.4, right=1)
plt.show()