#Çalışanların deneyim yılı ve maaş bilgileri verilmiştir.

pdf = """5 600
7 900
3 550
3 500
2 400
7 950
3 540
10 1200
6 900
4 550
8 1100
1 460
1 400
9 1000
1 380"""

import pandas as pd
pd.options.display.max_columns=None
pd.options.display.width = 1000
def doc_to_dataframe(doc):
    data = pd.DataFrame()
    doc_list = doc.replace("\n"," ").split(" ")

    data["deneyim_yili"] = [year for i,year in enumerate(doc_list,0) if i % 2 == 0]
    data["maas"] = [maas for i, maas in enumerate(doc_list, 0) if i % 2 != 0]

    return data.astype(int)

df = doc_to_dataframe(pdf)

# 1-Verilen bias ve weight’e göre doğrusal regresyon model denklemini oluşturunuz. Bias = 275, Weight= 90 (y’ = b+wx)

Bias = 275
Weight = 90

def linear_model(bias,weight,data):
    return bias + weight * data

# 2-Oluşturduğunuz model denklemine göre tablodaki tüm deneyim yılları için maaş tahmini yapınız.
linear_model(Bias,Weight,df["deneyim_yili"])

# 3-Modelin başarısını ölçmek için MSE, RMSE, MAE skorlarını hesaplayınız

predictions = linear_model(Bias,Weight,df["deneyim_yili"])

def score_calculator(how,y_real,y_pred):
    y_real, y_pred = y_real, y_pred
    if how == "MSE":
        return ((y_real - y_pred)**2).sum() / len(y_real)

    elif how == "RMSE":
        return (((y_real - y_pred)**2).sum() / len(y_real)) ** (1/2)

    elif how == "MAE":
        return ((abs(y_real - y_pred)).sum() / len(y_real))

    else : raise Exception("Enter the MSE, RMSE or MAE as metric")

score_calculator("MAE",df["maas"],predictions)

# Tablodaki boşlukları doldurarak ilerleyebilir ve verilen hata değerlendirme metrikleri formülleri ile rmse, mse, mae değerlerini hesaplayabilirsiniz.



df["Maaş Tahmini (y')"] = linear_model(Bias,Weight,df["deneyim_yili"])

df["Hata (y-y')"] = df["maas"] - df["Maaş Tahmini (y')"]

df["Hata Kareleri"] = df["Hata (y-y')"] ** 2

df["Mutlak Hata (|y-y'|)"] = abs(df["Hata (y-y')"])

