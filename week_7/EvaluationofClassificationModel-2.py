# Görev 1:
# Müşterinin churn olup olmama durumunu tahminleyen bir sınıflandırma modeli oluşturulmuştur. 10 test verisi gözleminin gerçek değerleri ve modelin tahmin ettiği olasılık değerleri verilmiştir.
import pandas as pd



pdf="""1 1 0.7
2 1 0.8
3 1 0.65
4 1 0.9
5 1 0.45
6 1 0.5
7 0 0.55
8 0 0.35
9 0 0.4
10 0 0.25
"""

def doc_to_data(doc):
    churn = []
    churn_proba = []
    data = pd.DataFrame()
    for i, row in enumerate(pdf.split("\n"),0):
        if len(row.split(" ")) == 3:
            churn.append(int(row.split(" ")[1]))
            churn_proba.append(float(row.split(" ")[2]))

    data["churn"],data["churn_proba"] = churn,churn_proba
    return data

df = doc_to_data(pdf)

def proba_rate(threshold,probas):
    return [1 if proba >= threshold else 0 for proba in probas]

# - Eşik değerini 0.5 alarak confusion matrix oluşturunuz.

df["churn_with_threshold"] = proba_rate(0.5,df["churn_proba"])

def confision_matrix(real,predict):
    board = pd.DataFrame()
    TP = sum([1 for r,p in zip(real,predict) if (r == 1) & (p == 1)])
    FP = sum([1 for r,p in zip(real,predict) if (r == 0) & (p == 1)])
    TN = sum([1 for r,p in zip(real,predict) if (r == 0) & (p == 0)])
    FN = sum([1 for r,p in zip(real,predict) if (r == 1) & (p == 0)])
    board["Churn (1)"] = [TP,FN, TP+FN]
    board["Non-Churn(0)"] = [FP, TN, FP+TN]
    board["+"] = [TP+FP,FN+TN," "]
    board.index = board.columns
    return print(board.to_markdown())

confision_matrix(df["churn"],df["churn_with_threshold"])
# - Accuracy, Recall, Precision, F1 Skorlarını hesaplayınız
def score_calculate(real,predict):
    board = pd.DataFrame()
    TP = sum([1 for r,p in zip(real,predict) if (r == 1) & (p == 1)])
    FP = sum([1 for r,p in zip(real,predict) if (r == 0) & (p == 1)])
    TN = sum([1 for r,p in zip(real,predict) if (r == 0) & (p == 0)])
    FN = sum([1 for r,p in zip(real,predict) if (r == 1) & (p == 0)])
    print(TP,FP,TN,FN)
    board["Accuracy"] = [(TP + TN) / ( TP + FP + TN + FN)]
    board["Recall"] =  [TP / (TP + FN)]
    board["Precision"] = [ TP / (TP + FP)]
    board["F1 Skor"] =  2 * (board["Recall"] * board["Precision"]) / (board["Recall"] + board["Precision"])
    return print(board.to_markdown(index=False))

score_calculate(df["churn"],df["churn_with_threshold"])

# Görev 2:
# Banka üzerinden yapılan işlemler sırasında dolandırıcılık işlemlerinin yakalanması amacıyla sınıflandırma modeli oluşturulmuştur. %90.5 doğruluk oranı elde edilen modelin başarısı yeterli bulunup model canlıya alınmıştır. Ancak canlıya alındıktan sonra modelin çıktıları beklendiği gibi olmamış, iş birimi modelin başarısız olduğunu iletmiştir. Aşağıda modelin tahmin sonuçlarının karmaşıklık matriksi verilmiştir. Buna göre;

pdf="""Fraud (1) 5 5 10
Non-Fraud (0) 90 900 990"""


TP,FN,FP,TN = int(pdf.split()[2]),int(pdf.split()[3]),int(pdf.split()[7]),int(pdf.split()[8])


# - Accuracy, Recall, Precision, F1 Skorlarını hesaplayınız.

def calculate(TP,FP,FN,TN):
    board = pd.DataFrame()
    board["Accuracy"] = [(TP + TN) / ( TP + FP + TN + FN)]
    board["Recall"] =  [TP / (TP + FN)]
    board["Precision"] = [ TP / (TP + FP)]
    board["F1 Skor"] = 2 * (board["Recall"] * board["Precision"]) / (board["Recall"] + board["Precision"])
    return print(board.to_markdown(index=False))

calculate(TP,FP,FN,TN)

# Veri Bilimi ekibinin gözden kaçırdığı durum ne olabilir yorumlayınız.

# Veri Bilimi ekibi targetin yanı bir dolandırıclık olup olmadığını belirten değerin dengesiz dağıtıldığını görmemiş.
# Bu yüzden Acc değerleri yüksek çıksada, model nadir görülen dolandırıcılık değerini gözden kaçırmış, bunu Recall da görebiliriz.
