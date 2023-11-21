# Telco Churn Prediction

#SO
# İş Problemi
#Şirketi terk edecek müşterileri tahmin edebilecek bir makine öğrenmesi modeli geliştirilmesi beklenmektedir.

#QUE
# Veri Seti Hikayesi
#Telco müşteri kaybı verileri, üçüncü çeyrekte Kaliforniya'daki 7043 müşteriye ev telefonu ve İnternet hizmetleri sağlayan hayali bir telekom şirketi hakkında bilgi içerir. Hangi müşterilerin hizmetlerinden ayrıldığını, kaldığını veya hizmete kaydolduğunu gösterir.

#SO
# |------------------|----------------------------------------------------------------------------------------------------------|
# | Features         | Explanation                                                                                              |
# |------------------|----------------------------------------------------------------------------------------------------------|
# | CustomerId       | Müşteri İd’si                                                                                            |
# | Gender           | Cinsiyet                                                                                                 |
# | SeniorCitizen    | Müşterinin yaşlı olup olmadığı (1, 0)                                                                    |
# | Partner          | Müşterinin bir ortağı olup olmadığı (Evet, Hayır)                                                        |
# | Dependents       | Müşterinin bakmakla yükümlü olduğu kişiler olup olmadığı (Evet, Hayır                                    |
# | tenure           | Müşterinin şirkette kaldığı ay sayısı                                                                    |
# | PhoneService     | Müşterinin telefon hizmeti olup olmadığı (Evet, Hayır)                                                   |
# | MultipleLines    | Müşterinin birden fazla hattı olup olmadığı (Evet, Hayır, Telefon hizmeti yok)                           |
# | InternetService  | Müşterinin internet servis sağlayıcısı (DSL, Fiber optik, Hayır)                                         |
# | OnlineSecurity   | Müşterinin çevrimiçi güvenliğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)                      |
# | OnlineBackup     | Müşterinin online yedeğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)                            |
# | DeviceProtection | Müşterinin cihaz korumasına sahip olup olmadığı (Evet, Hayır, İnternet hizmeti yok)                      |
# | TechSupport      | Müşterinin teknik destek alıp almadığı (Evet, Hayır, İnternet hizmeti yok)                               |
# | StreamingTV      | Müşterinin TV yayını olup olmadığı (Evet, Hayır, İnternet hizmeti yok)                                   |
# | StreamingMovies  | Müşterinin film akışı olup olmadığı (Evet, Hayır, İnternet hizmeti yok)                                  |
# | Contract         | Müşterinin sözleşme süresi (Aydan aya, Bir yıl, İki yıl)                                                 |
# | PaperlessBilling | Müşterinin kağıtsız faturası olup olmadığı (Evet, Hayır)                                                 |
# | PaymentMethod    | Müşterinin ödeme yöntemi (Elektronik çek, Posta çeki, Banka havalesi (otomatik), Kredi kartı (otomatik)) |
# | MonthlyCharges   | Müşteriden aylık olarak tahsil edilen tutar                                                              |
# | TotalCharges     | Müşteriden tahsil edilen toplam tutar                                                                    |
# | Churn            | Müşterinin kullanıp kullanmadığı (Evet veya Hayır)                                                       |
# |------------------|----------------------------------------------------------------------------------------------------------|

# Görev 1 : Keşifçi Veri Analizi
import pandas as pd
import numpy as np
pd.options.display.max_columns=None
pd.options.display.width = 1000
# Adım 1: Numerik ve kategorik değişkenleri yakalayınız.
df = pd.read_csv("week_7_8_9/7/Telco-Customer-Churn.csv")
df.head()

df.nunique()
target = "Churn"

def find_col_dtypes(df, target, cat_th = 10):
    num_cols = df.select_dtypes("number").columns.to_list()
    cat_cols = df.select_dtypes("object").columns.to_list()

    cardinals = [col for col in cat_cols if df[col].nunique() == df.shape[0]]

    cat_but_num = [col for col in cat_cols if (df[col].nunique() > cat_th) & (col not in cardinals)]

    cat_cols = [col for col in df.columns if (df[col].nunique() < cat_th) & (col not in [target])]

    num_cols = [col for col in num_cols if (col not in cat_cols) & (col not in [target])] + cat_but_num


    return num_cols, cat_cols

num_cols, cat_cols = find_col_dtypes(df, target, 10)

# Adım 2: Gerekli düzenlemeleri yapınız. (Tip hatası olan değişkenler gibi)
df[num_cols].dtypes
df["TotalCharges"] = df["TotalCharges"].astype(float) #hata verdiğini görmek için burada çalıştırıdm, sonra dönüp tekrar çalıştırabilirsin

df["TotalCharges"] = df["TotalCharges"].replace(" ",np.NaN)

df[cat_cols].dtypes

df["SeniorCitizen"] = df["SeniorCitizen"].astype("object")

df[target] = df[target].map({"No":0,"Yes":1})
# Adım 3: Numerik ve kategorik değişkenlerin veri içindeki dağılımını gözlemleyiniz.
df[num_cols].describe().T

df[cat_cols].describe(include="object").T

# Adım 4: Kategorik değişkenler ile hedef değişken incelemesini yapınız.
for col in cat_cols:
    print(df.groupby(col).agg({"Churn":"mean"}).to_markdown(), end="\n\n")

# Adım 5: Aykırı gözlem var mı inceleyiniz.

def outliner_detector(df,cols,take_care_outliners=False,print_outliners=False, q_1=0.25, q_3=0.75):
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
        temp.loc[col, "Outliner"] = f"{(data[col].max() > up) | (low > data[col].min())}"
        if take_care_outliners:
            data.loc[data[col] > up,col] = round(up)
            data.loc[data[col] < low,col] = round(low)
    if take_care_outliners:
        if print_outliners: print(temp.to_markdown())
        return data
    if print_outliners: print(temp.to_markdown())

outliner_detector(df,num_cols,print_outliners=True,q_1=0.25,q_3=0.75)

# Adım 6: Eksik gözlem var mı inceleyiniz.
df.isnull().sum()

# Görev 2 : Feature Engineering
# Adım 1: Eksik ve aykırı gözlemler için gerekli işlemleri yapınız.

df["TotalCharges"] = df["TotalCharges"].fillna(0)

# Adım 2: Yeni değişkenler oluşturunuz.

montly_quantiles = df["MonthlyCharges"].quantile([0,0.25,0.50,0.75]).to_list()
montly_quantiles.append(np.inf)
montly_quantiles[0] = -np.inf
df["Segment"] =  pd.cut(df["MonthlyCharges"],bins=montly_quantiles, labels=["D","C","B","A"]).astype("object")

print(df.groupby(by="Segment").agg({"Churn":"mean"}).to_markdown())


# Adım 3: Encoding işlemlerini gerçekleştiriniz.
new_num_cols, new_cat_cols = find_col_dtypes(df,"Churn",cat_th=10)
new_df = pd.get_dummies(df,columns=new_cat_cols,drop_first=True,dtype=int)

# Adım 4: Numerik değişkenler için standartlaştırma yapınız.
from sklearn.preprocessing import StandardScaler
new_df[new_num_cols] = StandardScaler().fit_transform(new_df[new_num_cols])

# Görev 3 : Modelleme
# Adım 1: Sınıflandırma algoritmaları ile modeller kurup, accuracy skorlarını inceleyip. En iyi 4 modeli seçiniz.

X = new_df.drop(["customerID","Churn"],axis=1)
y = new_df["Churn"]
y.value_counts()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=16,stratify=y)



from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score


def models(modelclass, X_train, X_test, y_train, y_test):
    md = modelclass.fit(X_train, y_train)
    y_pred = md.predict_proba(X_test)[:,1]
    return roc_auc_score(y_test, y_pred)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier

models(GradientBoostingClassifier(), X_train, X_test, y_train, y_test)

pd.DataFrame(dict(zip(lgb.feature_name_,lgb.feature_importances_)).items(),columns=["Feature","Importance"]).sort_values(by="Importance",ascending=False)

# Adım 2: Seçtiğiniz modeller ile hiperparametre optimizasyonu gerçekleştirin ve bulduğunuz hiparparametreler ile modeli tekrar kurunuz.

import optuna
def objective_hb(trial):

    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'max_iter': trial.suggest_int('max_iter', 300, 1000),
        'max_depth': trial.suggest_int('max_depth', 1, 10),
        "random_state": trial.suggest_categorical('random_state', [42])
    }


    model_hb = HistGradientBoostingClassifier(**params)
    model_hb.fit(X_train, y_train)
    y_pred = model_hb.predict_proba(X_test)[:,1]
    return roc_auc_score(y_test,y_pred)

study_hb = optuna.create_study(direction='maximize')
optuna.logging.set_verbosity(optuna.logging.WARNING)
study_hb.optimize(objective_hb, n_trials=50,show_progress_bar=True)

print('Best parameters', study_hb.best_params)

hg = HistGradientBoostingClassifier(**study_hb.best_params).fit(X_train,y_train)
y_pred = hg.predict_proba(X_test)[:,1]
roc_auc_score(y_test,y_pred)

def conf_matrix(model, X_test, y_test):
    from sklearn.metrics import ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    ConfusionMatrixDisplay.from_estimator(model,X_test, y_test,labels=[1,0],display_labels=("False", "True"),cmap="RdPu")
    plt.title(f"ROC-AUC: {roc_auc_score(y_test,y_pred_proba):.3f}     "
              f"Precision: {precision_score(y_test,y_pred):.3f}\n"
              f"Recall: {recall_score(y_test,y_pred):.3f}      "
              f"F1 Score: {f1_score(y_test,y_pred):.3f}\n")
    plt.xticks(ticks=[0,1],labels=[1,0])
    plt.yticks(ticks=[0,1],labels=[1,0])
    plt.text(x=-0.03,y=0.9,s="FP", color="black")
    plt.text(x=-0.04,y=-0.1,s="TP", color="black")
    plt.text(x=0.95,y=-0.1,s="FN", color="black")
    plt.text(x=0.95,y=0.9,s="TN", color="white")
    plt.show()

conf_matrix(hg,X_test,y_test)




# LGBM
import optuna
def objective_lgb(trial):

    params = {
        'metric': trial.suggest_categorical('metric', ['binary_error']),
        'max_depth': trial.suggest_int('max_depth', 1, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.1, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.01, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 1.0),
        "seed" : trial.suggest_categorical('seed', [42]),
        'verbose': trial.suggest_categorical('verbose', [-1])
    }


    model_lgb = LGBMClassifier(**params)
    model_lgb.fit(X_train, y_train)
    y_pred = model_lgb.predict_proba(X_test)[:,1]
    return roc_auc_score(y_test,y_pred)

study_lgb = optuna.create_study(direction='maximize')
optuna.logging.set_verbosity(optuna.logging.WARNING)
study_lgb.optimize(objective_lgb, n_trials=50,show_progress_bar=True)

print('Best parameters', study_lgb.best_params)

lgb = LGBMClassifier(**study_lgb.best_params)
lgb.fit(X_train, y_train)
y_pred = lgb.predict_proba(X_test)[:,1]

print('Accuracy: ', roc_auc_score(y_test, y_pred))


