
#QUE
# Feature Engineering
# İş Problemi


#Özellikleri belirtildiğinde kişilerin diyabet hastası olup olmadıklarını tahmin edebilecek bir makine öğrenmesi modeli geliştirilmesi istenmektedir.
# Modeli geliştirmeden önce gerekli olan veri analizi ve özellik mühendisliği adımlarını gerçekleştirmeniz beklenmektedir.

#Veri seti ABD'deki Ulusal Diyabet-Sindirim-Böbrek Hastalıkları Enstitüleri'nde tutulan büyük veri setinin parçasıdır.
# ABD'deki Arizona Eyaleti'nin en büyük 5. şehri olan Phoenix şehrinde yaşayan 21 yaş ve üzerinde olan Pima Indian kadınları üzerinde yapılan diyabet araştırması için kullanılan verilerdir.
# Hedef değişken "outcome" olarak belirtilmiş olup; 1 diyabet test sonucunun pozitif oluşunu, 0 ise negatif oluşunu belirtmektedir
#SO
# Pregnancies Hamilelik sayısı
# Glucose Oral glikoz tolerans testinde 2 saatlik plazma glikoz konsantrasyonu
# Blood Pressure Kan Basıncı (Küçük tansiyon) (mm Hg)
# SkinThickness Cilt Kalınlığı
# Insulin 2 saatlik serum insülini (mu U/ml)
# DiabetesPedigreeFunction Fonksiyon (Oral glikoz tolerans testinde 2 saatlik plazma glikoz konsantrasyonu)
# BMI Vücut kitle endeksi agirlik / boy^2
# Age Yaş (yıl)
# Outcome Hastalığa sahip (1) ya da değil (0)

#QUE
# Proje Görevleri

# Görev 1 : Keşifçi Veri Analizi
# Adım 1: Genel resmi inceleyiniz.
import pandas as pd
import numpy as np
pd.options.display.max_columns=None
pd.options.display.max_rows=10
pd.options.display.float_format = '{:.3f}'.format
pd.options.display.width = 1000

df = pd.read_csv("week_6/diabetes.csv")

df.head()

df.describe([0.01,0.25,0.75,0.99]).T

df["Age_Cat"] = pd.cut(df["Age"],bins=[0,30,40,50,60,df["Age"].max()],labels=["Youngs","30s","40s","50s","Old"]).astype("object")

def take_a_look(df):
    if df.empty: return "Data Frame is Empty"
    print(pd.DataFrame({'Rows': [df.shape[0]], 'Columns': [df.shape[1]]}, index=["Shape"]).to_markdown())
    print("\n")
    print(pd.DataFrame(df.dtypes, columns=["Type"]).to_markdown())
    print("\n")
    print(pd.DataFrame(df.nunique(),columns=["Number of Uniques"]).to_markdown())
    print("\n")
    print(pd.DataFrame(df.isnull().sum(),columns=["NaN"]).to_markdown())
    print("\n")
    print(df.describe([0.01, 0.25, 0.75, 0.99]).T.to_markdown(numalign="right",floatfmt=".1f"))
take_a_look(df)
# Adım 2: Numerik ve kategorik değişkenleri yakalayınız.
def num_cat(df):
    num_cols = df.select_dtypes(include="number").columns.to_list()
    num_list = [col for col in df.columns if (df[col].nunique() > 10) & (col in num_cols)]

    cat_list = df.select_dtypes(include="object").columns.to_list()
    #cat_list += [col for col in df.columns if (df[col].nunique() < 10) & (col not in cat_list)]

    return num_list,cat_list
num_cat(df)
# Adım 3: Numerik ve kategorik değişkenlerin analizini yapınız.
take_a_look(df[num_cat(df)[0]])


df[num_cat(df)[1]].describe(include="object").T

# Adım 4: Hedef değişken analizi yapınız. (Kategorik değişkenlere göre hedef değişkenin ortalaması, hedef değişkene göre
# numerik değişkenlerin ortalaması)
df.groupby(num_cat(df)[1]).agg({"Outcome": "mean"})

df.groupby("Outcome")[num_cat(df)[0]].mean()

# Adım 5: Aykırı gözlem analizi yapınız.
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

def ecem_outliner_detecter(df,cols):
    temp = pd.DataFrame()
    data = df.copy()
    for col in cols:
        q1 = data[col].quantile(0.05)
        q3 = data[col].quantile(0.95)
        IQR = q3 - q1
        up = q3 + 1.5 * IQR
        low = q1 - 1.5 * IQR
        temp.loc[col, "Min"] = round(data[col].min())
        temp.loc[col, "Low_Limit"] = round(low)
        temp.loc[col, "Mean"] = round(data[col].mean())
        temp.loc[col, "Median"] = round(data[col].median())
        temp.loc[col,"Up_Limit"] = round(up)
        temp.loc[col, "Max"] = round(data[col].max())
        temp.loc[col, "Outliner"] = f"{(data[col].max() > up) | (low > data[col].min())}"
    print(temp.to_markdown())

ecem_outliner_detecter(df,num_cat(df)[0])

outliner_detector(df,num_cat(df)[0],print_outliners=True,q_1=0.05, q_3=0.95)
# Adım 6: Eksik gözlem analizi yapınız.

df.isnull().sum()[df.isnull().sum()>0]

# Adım 7: Korelasyon analizi yapınız.
import seaborn as sns
import matplotlib.pyplot as plt

corr=df[num_cat(df)[0]].corr(numeric_only=True)

mask = np.triu(np.ones_like(corr))
plt.figure(figsize=(10,10))
sns.heatmap(corr, annot=True, mask=mask)
plt.show()


# Görev 2 : Feature Engineering
# Veri setinde eksik gözlem bulunmamakta ama Glikoz, Insulin vb.
# Değişkenlerde 0 değeri içeren gözlem birimleri eksik değeri ifade ediyor olabilir.
# Örneğin; bir kişinin glikoz veya insulin değeri 0 olamayacaktır.
# Bu durumu dikkate alarak sıfır değerlerini ilgili değerlerde NaN olarak atama yapıp sonrasında eksik değerlere işlemleri uygulayabilirsiniz.
# Adım 1: Eksik ve aykırı değerler için gerekli işlemleri yapınız.
df[num_cat(df)[0][1:]] = df[num_cat(df)[0][1:]].where(df!=0)

df[num_cat(df)[0][1:]] = df[num_cat(df)[0][1:]].replace(0,np.NaN)

missing_ones = df.isnull().sum()[df.isnull().sum()>0].index

def fill_based_cat(data,columns,based_cat,metric):
    data = data.copy()
    for col in columns:
        data[col] = data[col].fillna(df.groupby(based_cat)[col].transform(metric))
    return data

df = fill_based_cat(df,missing_ones,based_cat="Age_Cat",metric="median")

df.groupby("Age_Cat")["Glucose"].agg("median")

df["Glucose"].median()

df = outliner_detecter(df,num_cat(df)[0],take_care_outliners=True)
# Adım 2: Yeni değişkenler oluşturunuz.
df["Glucose_Insulin_Ratio"] = df["Glucose"] / df["Insulin"]
df["Skin_BMI_Ratio"] = df["SkinThickness"] / df["BMI"]
df["General_Glucose"] = df["DiabetesPedigreeFunction"] * df["Glucose"]

# Adım 3: Encoding işlemlerini gerçekleştiriniz.

df = pd.get_dummies(df,num_cat(df)[1],drop_first=True)

# Adım 4: Numerik değişkenler için standartlaştırma yapınız.
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
df[num_cat(df)[0]] = ss.fit_transform(df[num_cat(df)[0]])

# Adım 5: Model oluşturunuz.
from sklearn.model_selection import train_test_split
x = df.drop("Outcome",axis=1)
y = df["Outcome"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

rf = RandomForestClassifier(random_state=42)
rf.fit(x_train,y_train)
y_pred = rf.predict(x_test)
accuracy_score(y_pred,y_test)


pd.DataFrame(dict(zip(rf.feature_names_in_,rf.feature_importances_)).items(), columns=["Feature","Value"]).sort_values(by="Value", ascending=False).head(10)

