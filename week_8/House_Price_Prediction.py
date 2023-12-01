#SO
# Ev Fiyat Tahmin Modeli

#QUE
# İş Problemi

#SO
# Her bir eve ait özelliklerin ve ev fiyatlarının bulunduğu veriseti kullanılarak,
# farklı tipteki evlerin fiyatlarına ilişkin bir makine öğrenmesi projesi
# gerçekleştirilmek istenmektedir.


#Veri Seti Hikayesi

# Ames, Lowa’daki konut evlerinden oluşan bu veri seti içerisinde 79 açıklayıcı değişken bulunduruyor. Kaggle üzerinde bir yarışması
# da bulunan projenin veri seti ve yarışma sayfasına aşağıdaki linkten ulaşabilirsiniz. Veri seti bir kaggle yarışmasına ait
# olduğundan dolayı train ve test olmak üzere iki farklı csv dosyası vardır. Test veri setinde ev fiyatları boş bırakılmış olup, bu
# değerleri sizin tahmin etmeniz beklenmektedir

# Toplam Gözlem: 1460
# Sayısal Değişken: 38
# Kategorik Değişken: 43



# Görev 1: Keşifçi Veri Analizi
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.options.display.max_columns=100
pd.options.display.max_rows=100
pd.options.display.width=1000

# Adım 1: Train ve Test veri setlerini okutup birleştiriniz. Birleştirdiğiniz veri üzerinden ilerleyiniz.

path = "week_7_8_9/week_8/"
train = pd.read_csv(path+"train.csv").drop("Id",axis=1)
test = pd.read_csv(path+"test.csv").drop("Id",axis=1)
sub = pd.read_csv(path+"sample_submission.csv")

with open (path + "data_description.txt", "r") as file:
    desc = file.readlines()

with open (path + "data_turkish.txt", "r", encoding="utf8") as file:
    desc_turkish = file.readlines()

def what(question):
    import re
    row = re.compile(f"{question}.*?\n").search(" ".join(desc))
    if row:
        row = row.group(0)
        return print(row)
    else:
        return print(f"{question} couldn't find in description, check it please..", end="\n\n")

def nedir(question):
    import re
    row = re.compile(f"{question}.*?\n").search(" ".join(desc_turkish))
    if row:
        row = row.group(0)
        return print(row)
    else:
        return print(f"{question} couldn't find in description, check it please..", end="\n\n")

def con_cat(train, test):
    df1, df2 = train.copy(), test.copy()
    df1["group"] = "train"
    df2["group"] = "test"

    return pd.concat([df1, df2], axis=0, ignore_index=True)

df = con_cat(train, test)
df.head()
df.tail()
# Adım 2: Numerik ve kategorik değişkenleri yakalayınız.
def find_col_dtypes(data, ord_th):
    num_cols = data.select_dtypes("number").columns.to_list()
    cat_cols = data.select_dtypes("object").columns.to_list()

    ordinals = [col for col in num_cols if data[col].nunique() < ord_th]

    num_cols = [col for col in num_cols if col not in ordinals]

    return num_cols, ordinals, cat_cols

df.info()


num_cols, ordinals, cat_cols = find_col_dtypes(test, 20)

print(f"Num Cols: {num_cols}", end="\n\n")
print(f"Cat Cols: {cat_cols}", end="\n\n")
print(f"Ordinal Cols: {ordinals}")

# Adım 3: Gerekli düzenlemeleri yapınız. (Tip hatası olan değişkenler gibi)
df[num_cols].head()
df[ordinals].head()
df[cat_cols].head()

# Adım 4: Numerik ve kategorik değişkenlerin veri içindeki dağılımını gözlemleyiniz.
df[num_cols].describe([0.01,0.25,0.75,0.99]).T
df[cat_cols].describe(include="object").T

# Adım 5: Kategorik değişkenler ile hedef değişken incelemesini yapınız.
target = "SalePrice"
plt.figure(figsize=(14,len(cat_cols)*3))
for idx,column in enumerate(cat_cols):
    data = df[df["group"] == "train"].groupby(column)[target].agg(["mean", "count"]).reset_index().sort_values(by="count")
    plt.subplot(len(cat_cols)//2+1,2,idx+1)
    sns.barplot(y=column, x="count", data=data, palette="pastel")
    for i, val in enumerate(data.values, 0):
        plt.text(val[2] + 10, i + 0.05 / len(data), f"{int(val[1] // 1000)}k", color='black', fontsize=11)
    plt.title(f"{column}: Count and Mean: {target}")
    plt.xticks(fontweight='bold')
    plt.box(False)
    plt.tight_layout()
plt.show()

for col in cat_cols :
   print(df[df["group"] == "train"].groupby(col).agg(mean=(target,"mean"),
                                                     count=(col,"count"),
                                                     percent=(col,lambda x: f"{(len(x) / len(train) * 100):.1f}%"))\
         .sort_values(by="count").to_markdown(), end="\n\n")

# Adım 6: Aykırı gözlem var mı inceleyiniz.

def outliner_detector(df, cols, take_care_outliners=False, print_outliners=False, q_1=0.25, q_3=0.75):
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
        temp.loc[col, "Outliner"] = "Min-Max-Outliner" if (data[col].max() > up) & (low > data[col].min())\
                                    else ("Max-Outliner" if data[col].max() > up \
                                    else ("Min-Outliner" if low > data[col].min() \
                                    else "No"))
        if take_care_outliners:
            data.loc[data[col] > up,col] = round(up-1)
            data.loc[data[col] < low,col] = round(low-1)
    if take_care_outliners:
        if print_outliners: return temp
        return data
    if print_outliners: return temp

outliner_detector(df, num_cols, print_outliners=True, q_1=0.01, q_3=0.99)

# Adım 7: Eksik gözlem var mı inceleyiniz.
df[num_cols].isnull().sum()
df[cat_cols].isnull().sum()
df[ordinals].isnull().sum()



# Görev 2: Feature Engineering

# Adım 1: Eksik ve aykırı gözlemler için gerekli işlemleri yapınız.
df[num_cols].isnull().sum()
nedir("LotFrontage")


def mice_imput(df: pd.DataFrame, fill: str, based: list) -> pd.Series:
    """
    Impute missing values in a specified column of a DataFrame using the
    MICE (Multiple Imputation by Chained Equations) algorithm.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - fill (str): The column name with missing values to be imputed.
    - based (list): A list of column names considered as features for imputation.

    Returns:
    - pd.Series: A Series containing the imputed values for the specified column.

    MICE (Multiple Imputation by Chained Equations) is a statistical method used for imputing
    missing data in a dataset.
    It is an iterative algorithm that imputes missing values one variable at a time,
    considering the relationships between variables. In this implementation:

    1. Categorical columns are identified in the 'based' list.
    2. A temporary DataFrame is created by one-hot encoding categorical columns and
        selecting the target column ('fill').
    3. A missing value mask is generated for the temporary DataFrame.
    4. The IterativeImputer from scikit-learn is used to impute missing values iteratively.
    5. The imputed values are assigned to the original DataFrame in the specified column.
    """

    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer

    categoric_cols = [col for col in based if df[col].dtype == "O"]

    temp_df = pd.get_dummies(df[[fill] + based].copy(), columns=categoric_cols)

    missing_mask = temp_df.isna()

    imputer = IterativeImputer(max_iter=10, random_state=42)

    imputed_values = imputer.fit_transform(temp_df)

    temp_df[fill][temp_df[fill].isnull()] = imputed_values[missing_mask]

    return temp_df[fill]

df["LotFrontage"] = mice_imput(df, fill="LotFrontage", based=["LotArea","LotShape","LotConfig"])

df[num_cols].isnull().sum()

what("MasVnrArea")
df.loc[df["MasVnrArea"].isnull()].head(5)

df.loc[df["MasVnrArea"].isnull(),["MasVnrArea"]] = 0

df[num_cols].isnull().sum()
what("BsmtFinSF1")

df.loc[df["BsmtFinSF1"].isnull(),["BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF"]]

df.loc[df["BsmtFinSF1"].isnull(),["BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF"]] = 0

df.loc[[2120]]

df[num_cols].isnull().sum()

df[df["GarageYrBlt"].isnull()]

df.loc[df["GarageYrBlt"].isnull(),"GarageYrBlt"] = 0

df[num_cols].isnull().sum()
df.loc[df["GarageArea"].isnull(),["GarageYrBlt","GarageArea"]]

df.loc[df["GarageArea"].isnull(),"GarageArea"] = 0


df[cat_cols].isnull().sum()

df[cat_cols] = df[cat_cols].fillna("None")

df[cat_cols].isnull().sum().sum()

df[ordinals].isnull().sum()

df[ordinals] = df[ordinals].fillna(0)

df = outliner_detector(df, num_cols, take_care_outliners= True, q_1=0.01, q_3=0.99)

outliner_detector(df, num_cols, print_outliners=True, q_1=0.01, q_3=0.99)


for col in cat_cols :
   print(df.groupby(col).agg(count=(col,"count"),
                            percent=(col,lambda x: f"{(len(x) / len(df) * 100):.1f}%")).sort_values(by="count").to_markdown(), end="\n\n")

# Adım 2: Rare Encoder uygulayınız.

#dominant
df["Street"].value_counts() / len(df) * 100
def drop_feature(data,columns, percentage):
    data = data.copy()
    new_cat_cols=[]
    for col in columns:
        rank_1 = (data[col].value_counts().sort_values(ascending=False) / len(data)*100).iloc[0]
        if rank_1 > percentage:
            print(f"Feature {col} is Nonsense, Dropped")
            data.drop(col, axis=1, inplace=True)
        else:
            new_cat_cols.append(col)
    return data, new_cat_cols

df, new_cat_cols = drop_feature(df,cat_cols,percentage = 95)

#rare

df["HeatingQC"].value_counts() / len(df) * 100
df["SaleCondition"].value_counts() / len(df) * 100
def bag_rares(data, columns, percentage):
    data = data.copy()
    for col in columns:
        rares = data[col].value_counts().sort_values(ascending=False) / len(df) < percentage/100
        rare_names = rares[rares].index.to_list()
        data[col][data[col].isin(rare_names)] = "Rare"
    return data

df = bag_rares(df,new_cat_cols,percentage = 5)

df["SaleCondition"].value_counts() / len(df) * 100
df["Neighborhood"].value_counts() / len(df) * 100


for col in new_cat_cols :
   print(df.groupby(col).agg(count=(col,"count"),
                            percent=(col,lambda x: f"{(len(x) / len(df) * 100):.1f}%")).sort_values(by="count").to_markdown(), end="\n\n")
# Adım 3: Yeni değişkenler oluşturunuz.

def new_features(df):
    # Calculate the total area
    df['TotalArea'] = df['TotalBsmtSF'] + df['GrLivArea']

    # Calculate the number of new or renovated bathrooms
    df['TotalBathrooms'] = df['FullBath'] + df['HalfBath']*0.5 + df["BsmtHalfBath"]*0.5 + df["BsmtFullBath"]

    # Calculate the total room count
    df['TotalRooms'] = df['BedroomAbvGr'] + df['TotRmsAbvGrd']

    # Calculate the total porch area
    df['TotalPorchArea'] = df['OpenPorchSF'] + df['EnclosedPorch'] + df["3SsnPorch"] + df["ScreenPorch"] + df["WoodDeckSF"]

    # House Overall
    df['Overal'] = df['OverallQual'] + df['OverallCond']

    # Has pool?
    df['HasPool'] = [1 if pool > 0 else 0 for pool in df["PoolArea"]]


new_features(df)

df["HasPool"].value_counts() / len(df) * 100
# Adım 4: Encoding işlemlerini gerçekleştiriniz
df = pd.get_dummies(df, columns=new_cat_cols, dtype=int)

df.head()
# Görev 3: Model Kurma

# Adım 1: Train ve Test verisini ayırınız. SalePrice değişkeni boş olan değerler test verisidir :D
train = df[df["group"] == "train"].drop("group", axis = 1)
test = df[df["group"] == "test"].drop(["group","SalePrice"], axis = 1)
train.columns = [col.replace(" ", "_") for col in train.columns]
test.columns = [col.replace(" ", "_") for col in test.columns]
# Adım 2: Train verisi ile model kurup, model başarısını değerlendiriniz.
# Bonus: Hedef değişkene log dönüşümü yaparak model kurunuz ve rmse sonuçlarını gözlemleyiniz. Not: Log'un tersini (inverse)
# almayı unutmayınız.
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
train["SalePrice"].plot(kind="kde")
plt.show()

X = train.drop(["SalePrice"], axis=1)
y = np.log(train["SalePrice"])

y.plot(kind="kde")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state = 42)

import lightgbm
lgb = lightgbm.LGBMRegressor(objective = 'root_mean_squared_error', verbose=-1)
lgb.fit(X_train, y_train)
mean_squared_error(y_test,lgb.predict(X_test), squared=False)

lightgbm.plot_importance(lgb, max_num_features = 15);
plt.show()

import xgboost
xgb = xgboost.XGBRegressor(objective = 'reg:squarederror')
xgb.fit(X_train, y_train)
mean_squared_error(y_test,xgb.predict(X_test), squared=False)

xgboost.plot_importance(xgb, max_num_features = 15);
plt.show()

from sklearn.preprocessing import MinMaxScaler
lgb_importances = pd.DataFrame(dict(lgbm = lgb.feature_importances_), index=lgb.feature_name_)
xgb_importances = pd.DataFrame(dict(xgb = xgb.feature_importances_), index=xgb.feature_names_in_)
importances = pd.concat([lgb_importances,xgb_importances],axis=1)
min_max = MinMaxScaler((0.1,1))
importances["cross"] = min_max.fit_transform(importances[["lgbm"]]) * min_max.fit_transform(importances[["xgb"]])
sorted = importances.sort_values(by="cross", ascending=False).reset_index()
sorted

sorted.tail(101)

def drop_calculate():
    attempts = {}
    best_score = 0.5
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    from tqdm.auto import tqdm
    for i in tqdm(range(1, len(X_train.columns))):
        drop_col = sorted.iloc[len(sorted) - i]["index"]
        X_train.drop(drop_col, axis=1, inplace=True)
        X_test.drop(drop_col, axis=1, inplace=True)

        lgb.fit(X_train, y_train)
        y_pred = lgb.predict(X_test)
        score = mean_squared_error(y_test, y_pred, squared=False)

        attempts[i] = score

        if score < best_score:
            best_score = score
    return pd.DataFrame(attempts.values(), index = attempts.keys(), columns=['Results']).sort_values(by = "Results", ascending=True).head(10)
drop_calculate()

X_train.drop(sorted.tail(41)["index"],axis=1, inplace=True)
X_test.drop(sorted.tail(41)["index"],axis=1, inplace=True)
test.drop(sorted.tail(41)["index"],axis=1, inplace=True)
# Adım 3: Hiperparemetre optimizasyonu gerçekleştiriniz.

from lightgbm import LGBMRegressor
import optuna

def objective_lgb(trial):
    """Define the objective function"""

    params = {
        'objective': trial.suggest_categorical('objective', ['root_mean_squared_error']), #refers to the main goal that the model is trying to optimize (most important)
        'max_depth': trial.suggest_int('max_depth', 3, 10), #limit the max depth for tree model. This is used to deal with over-fitting Tree still grows leaf-wise
        'num_leaves': trial.suggest_int('num_leaves', 3, 32), #max number of leaves in one tree
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 30), #minimal number of data in one leaf. Can be used to deal with over-fitting
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1), #shrinkage rate
        'n_estimators': trial.suggest_int('n_estimators', 300, 700), # number of boosting iterations
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.1, 1), # like feature_fraction, but this will randomly select part of data without resampling
        'feature_fraction': trial.suggest_float('feature_fraction', 0.1, 1), # if you set it to 0.8, LightGBM will select 80% of features before training each tree
        'feature_fraction_bynode': trial.suggest_float('feature_fraction_bynode', 0.1, 1), # if you set it to 0.8, LightGBM will select 80% of features at each tree node
        'lambda_l1': trial.suggest_float('lambda_l1', 0, 1), #L1 (Lasso) regularization Can be used to deal with over-fitting,
        'lambda_l2': trial.suggest_float('lambda_l2', 0, 1), #L2 (Ridge) regularization Can be used to deal with over-fitting
        "random_state" : trial.suggest_categorical('random_state', [42]), #this seed is used to generate other seeds
        'verbose': trial.suggest_categorical('verbose', [-1]), #controls the level of LightGBM’s verbosity < 0: Fatal, = 0: Error (Warning), = 1: Info, > 1: Debug
    }


    model_lgb = LGBMRegressor(**params)
    model_lgb.fit(X_train, y_train)
    y_pred = model_lgb.predict(X_test)
    return mean_squared_error(y_test,y_pred, squared=False)

study_lgb = optuna.create_study(direction='minimize')
optuna.logging.set_verbosity(optuna.logging.WARNING)
study_lgb.optimize(objective_lgb, n_trials=100,show_progress_bar=True)

# Print the best parameters
study_lgb.best_params

optuna.visualization.matplotlib.plot_slice(study_lgb, params=['max_depth', 'num_leaves', 'min_data_in_leaf',"learning_rate"])
optuna.visualization.matplotlib.plot_slice(study_lgb, params=['n_estimators', 'bagging_fraction', 'feature_fraction',"feature_fraction_bynode"])
optuna.visualization.matplotlib.plot_slice(study_lgb, params=['lambda_l1', 'lambda_l2'])


plt.show()

lgb = LGBMRegressor(**study_lgb.best_params)
lgb.fit(X_train, y_train)
y_pred = lgb.predict(X_test)

print('Error: ', mean_squared_error(y_test,y_pred, squared=False))

# Adım 4: Değişken önem düzeyini inceleyeniz.
lightgbm.plot_importance(lgb, max_num_features = 30);
plt.show()

pd.DataFrame(dict(lgbm = lgb.feature_importances_), index=lgb.feature_name_).sort_values(by="lgbm", ascending=False).head(20)
# Bonus: Test verisinde boş olan salePrice değişkenlerini tahminleyiniz ve Kaggle sayfasına submit etmeye uygun halde bir
# dataframe oluşturup sonucunuzu yükleyiniz.

sub["SalePrice"]=np.exp(lgb.predict(test))
sub.to_csv(path+'submission.csv',index=False)
sub