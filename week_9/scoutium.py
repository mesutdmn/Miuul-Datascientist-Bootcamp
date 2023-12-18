# Makine Öğrenmesi ile Yetenek Avcılığı Sınıflandırma
#
# İş Problemi
# Scout’lar tarafından izlenen futbolcuların özelliklerine verilen puanlara göre, oyuncuların hangi sınıf
# (average, highlighted) oyuncu olduğunu tahminleme
#
# Scoutium: futbol ve diğer sporlarda yetenekli genç oyuncuları keşfetmek ve oyuncu transferi yapmak amacıyla kurulmuş bir platformdur.
# Scout: sporlarda yetenek avcılarını veya oyuncu izleyicilerini ifade eder

# Veri Seti Hikayesi
# Veri seti Scoutium’dan maçlarda gözlemlenen futbolcuların özelliklerine göre scoutların değerlendirdikleri futbolcuların, maç
# içerisinde puanlanan özellikleri ve puanlarını içeren bilgilerden oluşmaktadır.
#
# scoutium_attributes.csv

# task_response_id: Bir scoutun bir maçta bir takımın kadrosundaki tüm oyunculara dair değerlendirmelerinin kümesi
# match_id: İlgili maçın id'si
# evaluator_id: Değerlendiricinin(scout'un) id'si
# player_id: İlgili oyuncunun id'si
# position_id: İlgili oyuncunun o maçta oynadığı pozisyonun id’si
#                                                                 1: Kaleci
#                                                                 2: Stoper
#                                                                 3: Sağ bek
#                                                                 4: Sol bek
#                                                                 5: Defansif orta saha
#                                                                 6: Merkez orta saha
#                                                                 7: Sağ kanat
#                                                                 8: Sol kanat
#                                                                 9: Ofansif orta saha
#                                                                 10: Forvet
# analysis_id: Bir scoutun bir maçta bir oyuncuya dair özellik değerlendirmelerini içeren küme
# attribute_id: Oyuncuların değerlendirildiği her bir özelliğin id'si
# attribute_value: Bir scoutun bir oyuncunun bir özelliğine verdiği değer(puan)

# scoutium_potential_labels.csv

# task_response_id: Bir scoutun bir maçta bir takımın kadrosundaki tüm oyunculara dair değerlendirmelerinin kümesi
# match_id İlgili: maçın id'si
# evaluator_id: Değerlendiricinin(scout'un) id'si
# player_id: İlgili oyuncunun id'si
# potential_label: Bir scoutun bir maçta bir oyuncuyla ilgili nihai kararını belirten etiket. (hedef değişken)

# Görevler
# Adım 1: scoutium_attributes.csv ve scoutium_potential_labels.csv dosyalarını okutunuz.
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import colorama

YEL = colorama.Fore.LIGHTYELLOW_EX
colorama.init(autoreset=True)
warnings.filterwarnings("ignore")
pd.options.display.max_columns=100
pd.options.display.max_rows=20
pd.options.display.width=1000

att = pd.read_csv("week_7_8_9/week_9/scoutium_attributes.csv", sep=";")
pot = pd.read_csv("week_7_8_9/week_9/scoutium_potential_labels.csv", sep=";")

# Adım 2: Okutmuş olduğumuz csv dosyalarını merge fonksiyonunu kullanarak birleştiriniz.
# ("task_response_id", 'match_id', 'evaluator_id' "player_id" 4 adet değişken üzerinden birleştirme işlemini gerçekleştiriniz.)

df = att.merge(pot, on=["task_response_id", 'match_id', 'evaluator_id', "player_id"])

# Adım 3: position_id içerisindeki Kaleci (1) sınıfını veri setinden kaldırınız.
df = df[df["position_id"]!=1]

# Adım 4: potential_label içerisindeki below_average sınıfını veri setinden kaldırınız.( below_average sınıfı tüm verisetinin %1'ini oluşturur)
df = df[df["potential_label"]!="below_average"]
# Adım 5: Oluşturduğunuz veri setinden “pivot_table” fonksiyonunu kullanarak bir tablo oluşturunuz. Bu pivot table'da her satırda bir oyuncu
# olacak şekilde manipülasyon yapınız.
# Adım 1: İndekste “player_id”,“position_id” ve “potential_label”, sütunlarda “attribute_id” ve değerlerde scout’ların oyunculara verdiği puan
# “attribute_value” olacak şekilde pivot table’ı oluşturunuz

# |:---------:|:-----------:|:---------------:|:------:|:------:|:------:|:------:|:------:|:------:|:--:|
# |           |             |                 |  4322  |  4323  |  4324  |  4325  |  4325  |  4327  | .. |
# |:---------:|:-----------:|:---------------:|:------:|:------:|:------:|:------:|:------:|:------:|:--:|
# | player_id | position_id | potential_label |        |        |        |        |        |        | .. |
# |  1355710  |      7      |     average     | 50.500 | 50.500 | 34.000 | 50.500 | 45.000 | 45.000 | .. |
# |  1356362  |      9      |     average     | 67.000 | 67.000 | 67.000 | 67.000 | 67.000 | 67.000 | .. |
# |  1356375  |      3      |     average     | 67.000 | 67.000 | 67.000 | 67.000 | 67.000 | 67.000 | .. |
# |           |      4      |     average     | 67.000 | 78.000 | 67.000 | 67.000 | 67.000 | 78.000 | .. |
# |  1356411  |      9      |     average     | 67.000 | 67.000 | 78.000 | 78.000 | 67.000 | 67.000 | .. |
# |:---------:|:-----------:|:---------------:|:------:|:------:|:------:|:------:|:------:|:------:|:--:|
pivot = df.pivot_table(index=["player_id","position_id","potential_label"],columns="attribute_id", values="attribute_value")

pivot = df.groupby(["player_id","position_id","potential_label","attribute_id"])["attribute_value"].mean().unstack()
# Adım 2: “reset_index” fonksiyonunu kullanarak indeksleri değişken olarak atayınız ve “attribute_id” sütunlarının isimlerini stringe çeviriniz.
pivot = pivot.reset_index()
pivot.columns = [str(col) for col in pivot.columns]

# Adım 6: Label Encoder fonksiyonunu kullanarak “potential_label” kategorilerini (average, highlighted) sayısal olarak ifade ediniz.
pivot["potential_label"].value_counts()
pivot["potential_label"] = pivot["potential_label"].map({"average":0,"highlighted":1})
# Adım 7: Sayısal değişken kolonlarını “num_cols” adıyla bir listeye atayınız.
num_cols = [col for col in pivot.columns if pivot[col].dtype == "float64"]
len(num_cols)
pivot.dtypes
# Adım 8: Kaydettiğiniz bütün “num_cols” değişkenlerindeki veriyi ölçeklendirmek için StandardScaler uygulayınız.
from sklearn.preprocessing import StandardScaler
pivot[num_cols] = StandardScaler().fit_transform(pivot[num_cols])
# Adım 9: Elimizdeki veri seti üzerinden minimum hata ile futbolcuların potansiyel etiketlerini tahmin eden bir makine öğrenmesi modeli
# geliştiriniz. (Roc_auc, f1, precision, recall, accuracy metriklerini yazdırınız.)
#models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
#metrics
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score



def scores(model, X_test, y_test):
    y_predict_proba = model.predict_proba(X_test)[:,1]
    y_pred = model.predict(X_test)
    print(f"{YEL}{type(model).__name__}")
    print(f"\t-> Roc_Auc_Score: {roc_auc_score(y_test,y_predict_proba):.2f}\n"
          f"\t-> F1_Score: {f1_score(y_test,y_pred):.2f}\n"
          f"\t-> Precision_Score: {precision_score(y_test,y_pred):.2f}\n"
          f"\t-> Recall_Score: {recall_score(y_test,y_pred):.2f}\n"
          f"\t-> Accuracy_Score: {accuracy_score(y_test,y_pred):.2f}\n")
def base_models(X_train, y_train):

    models = [LogisticRegression(),DecisionTreeClassifier(),KNeighborsClassifier(),
              RandomForestClassifier(), GradientBoostingClassifier(),
              HistGradientBoostingClassifier(),LGBMClassifier(verbose=-1),XGBClassifier()]
    for model in models:
        model.fit(X_train, y_train)
    for model in models:
        scores(model, X_test, y_test)


from sklearn.model_selection import train_test_split
X = pivot[["position_id"]+num_cols]
y = pivot["potential_label"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

ss = StandardScaler()
X_train[num_cols] = ss.fit_transform(X_train[num_cols])
X_test[num_cols] = ss.transform(X_test[num_cols])


base_models(X_train, y_train)

import lightgbm
lgb = lightgbm.LGBMClassifier(verbose=-1)
lgb.fit(X_train, y_train)
scores(lgb, X_test, y_test)

# Adım 10: Değişkenlerin önem düzeyini belirten feature_importance fonksiyonunu kullanarak özelliklerin sıralamasını çizdiriniz.
plt.style.use("week_7_8_9/week_9/cyberpunk.mplstyle")
lightgbm.plot_importance(lgb)
plt.show()

# matlotlib ayarlarını resetlemek için.
import matplotlib
matplotlib.rc_file_defaults()
