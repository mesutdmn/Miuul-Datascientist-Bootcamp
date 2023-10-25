#####################################################
# AB Testi ile BiddingYöntemlerinin Dönüşümünün Karşılaştırılması
#####################################################

#####################################################
# İş Problemi
#####################################################

# Facebook kısa süre önce mevcut "maximumbidding" adı verilen teklif verme türüne alternatif
# olarak yeni bir teklif türü olan "average bidding"’i tanıttı. Müşterilerimizden biri olan bombabomba.com,
# bu yeni özelliği test etmeye karar verdi veaveragebidding'in maximumbidding'den daha fazla dönüşüm
# getirip getirmediğini anlamak için bir A/B testi yapmak istiyor.A/B testi 1 aydır devam ediyor ve
# bombabomba.com şimdi sizden bu A/B testinin sonuçlarını analiz etmenizi bekliyor.Bombabomba.com için
# nihai başarı ölçütü Purchase'dır. Bu nedenle, istatistiksel testler için Purchasemetriğine odaklanılmalıdır.




#####################################################
# Veri Seti Hikayesi
#####################################################

# Bir firmanın web site bilgilerini içeren bu veri setinde kullanıcıların gördükleri ve tıkladıkları
# reklam sayıları gibi bilgilerin yanı sıra buradan gelen kazanç bilgileri yer almaktadır.Kontrol ve Test
# grubu olmak üzere iki ayrı veri seti vardır. Bu veri setleriab_testing.xlsxexcel’ininayrı sayfalarında yer
# almaktadır. Kontrol grubuna Maximum Bidding, test grubuna AverageBiddinguygulanmıştır.

# impression: Reklam görüntüleme sayısı
# Click: Görüntülenen reklama tıklama sayısı
# Purchase: Tıklanan reklamlar sonrası satın alınan ürün sayısı
# Earning: Satın alınan ürünler sonrası elde edilen kazanç



#####################################################
# Proje Görevleri
#####################################################

######################################################
# AB Testing (Bağımsız İki Örneklem T Testi)
######################################################

# 1. Hipotezleri Kur
# 2. Varsayım Kontrolü
#   - 1. Normallik Varsayımı (shapiro)
#   - 2. Varyans Homojenliği (levene)
# 3. Hipotezin Uygulanması
#   - 1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi
#   - 2. Varsayımlar sağlanmıyorsa mannwhitneyu testi
# 4. p-value değerine göre sonuçları yorumla
# Not:
# - Normallik sağlanmıyorsa direkt 2 numara. Varyans homojenliği sağlanmıyorsa 1 numaraya arguman girilir.
# - Normallik incelemesi öncesi aykırı değer incelemesi ve düzeltmesi yapmak faydalı olabilir.




#####################################################
# Görev 1:  Veriyi Hazırlama ve Analiz Etme
#####################################################
import pandas as pd
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu

import colorama
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.stats.api as sms
pd.options.display.max_columns=None
pd.options.display.max_rows=10
pd.options.display.float_format = '{:.3f}'.format
pd.options.display.width = 1000

YEL = colorama.Fore.LIGHTYELLOW_EX
BLU = colorama.Fore.LIGHTBLUE_EX
colorama.init(autoreset=True)

# Adım 1:  ab_testing_data.xlsx adlı kontrol ve test grubu verilerinden oluşan veri setini okutunuz. Kontrol ve test grubu verilerini ayrı değişkenlere atayınız.

pd.ExcelFile("week_4/ab_testing.xlsx").sheet_names

control = pd.read_excel("week_4/ab_testing.xlsx", sheet_name="Control Group")
test = pd.read_excel("week_4/ab_testing.xlsx", sheet_name="Test Group")

# Adım 2: Kontrol ve test grubu verilerini analiz ediniz.

control.describe([0.01,0.99]).T
test.describe([0.01,0.99]).T

# Adım 3: Analiz işleminden sonra concat metodunu kullanarak kontrol ve test grubu verilerini birleştiriniz.

control["Group"] = "control"
test["Group"] = "test"
df = pd.concat([control,test], ignore_index=True)
df

c,t = sms.DescrStatsW(df["Purchase"]).tconfint_mean()
print(f"{YEL} Güven Aralığı: {c: .3f} - {t: .3f}")

#####################################################
# Görev 2:  A/B Testinin Hipotezinin Tanımlanması
#####################################################

# Adım 1: Hipotezi tanımlayınız.

#QUE
# H₀: İki grup arasında istatistiksel olarak anlamlı bir fark yoktur.
# H₁: İki grup arasında istatistiksel olarak anlamlı bir fark vardır.

# Adım 2: Kontrol ve test grubu için purchase(satın alma) ortalamalarını analiz ediniz

c,t = control['Purchase'].mean(),test['Purchase'].mean()

print(f"{YEL}MaximumBidding Grubu Purchase Ortalaması: {c:.3f}")
print(f"{YEL}AverageBidding Grubu Purchase Ortalaması: {t:.3f}")


#####################################################
# GÖREV 3: Hipotez Testinin Gerçekleştirilmesi
#####################################################


# Adım 1: Hipotez testi yapılmadan önce varsayım kontrollerini yapınız.Bunlar Normallik Varsayımı ve Varyans Homojenliğidir.
# Kontrol ve test grubunun normallik varsayımına uyup uymadığını Purchase değişkeni üzerinden ayrı ayrı test ediniz

#QUE
# H₀: Control Grubundaki Gözlemler Normal Dağılmıştır
# H₁: Control Grubundaki Gözlemler Normal Dağılmamıştır

w,p = shapiro(control['Purchase'])
print(f"{YEL} W: {w: .3f} \n p-Value: {p: .3f}")
0.05
#QUE
# H₀: Test Grubundaki Gözlemler Normal Dağılmıştır
# H₁: Test Grubundaki Gözlemler Normal Dağılmamıştır
w,p = shapiro(test['Purchase'])
print(f"{YEL} W: {w: .3f} \n p-Value: {p: .3f}")

control['Purchase'].plot(kind="kde")
test['Purchase'].plot(kind="kde")
plt.show()




bozuk = control[['Purchase']].copy()
bozuk.loc[bozuk['Purchase']<800]=0
w,p = shapiro(bozuk['Purchase'])
print(f"{YEL} W: {w: .3f} \n p-Value: {p: .14f}")

bozuk['Purchase'].plot(kind="kde")
plt.show()

#TODO
# varyans homojenliği  (benzerliği)

#QUE
# H₀: Control ve Test gruplarının varyansları arasında anlamlı bir farklılık yoktur.
# H₁: Control ve Test gruplarının varyansları arasında anlamlı bir farklılık vardır.

w,p = levene(control['Purchase'],test['Purchase'])
print(f"{YEL}W: {w:.3f}\np-Value: {p: .3f}")

#SO:
# p-value değeri 0.1082 olduğu için, varyans benzerliği için null hipotezi reddedilemez Yani, grupların varyansları arasında anlamlı bir farklılık olduğuna dair yeterince güçlü bir kanıt elde edilememiştir. (Varyansları Benzer)

sns.boxplot(y="Purchase", x="Group", data=df, showmeans=True, palette="pastel")
plt.show()
sns.histplot(x="Purchase", hue="Group", data=df, palette="colorblind")
plt.show()


levene(control['Purchase'],bozuk['Purchase'])
bozuk["Group"] = "bozuk"
df2 = pd.concat([control,bozuk], ignore_index=True)
sns.histplot(x="Purchase", hue="Group", data=df2, palette="colorblind")
plt.show()
# Adım 2: Normallik Varsayımı ve Varyans Homojenliği sonuçlarına göre uygun testi seçiniz

#SO:
# Normal dağılım hipotezi red edilmediğinden parametik bir yöntem kullanmamız gerekmektedir.

#QUE
# H₀: İki grup ortalaması arasında istatistiksel olarak anlamlı bir fark yoktur.
# H₁: İki grup ortalaması arasında istatistiksel olarak anlamlı bir fark vardır.

t, p, = ttest_ind(control['Purchase'], test['Purchase'], equal_var = True)
print(f"{YEL}\tt:  {t:.3f}\n\tp-Value: {p:.3f}")


# Adım 3: Test sonucunda elde edilen p_value değerini göz önünde bulundurarak kontrol ve test grubu satın alma
# ortalamaları arasında istatistiki olarak anlamlı bir fark olup olmadığını yorumlayınız.

#SO
# p-value 0.349 çıktı, Buda iki grup ortalaması arasında istatistiksel olarak anlamlı bir fark yoktur, yani H₀ hipotezi red edilemez.

##############################################################
# GÖREV 4 : Sonuçların Analizi
##############################################################

# Adım 1: Hangi testi kullandınız, sebeplerini belirtiniz.

#SO:
# Normallik hipotezini red edemediğimiz için ttest_ind (bağımsız iki örneklem T-Testi) kullandım, varyans benzerliğinide red edemediğimiz için equal_var parametresini True girdik.
# varyansları benzer olduğundan "Student's t-test" kullandık, eşit olmasaydı "Welch's t-test" kullancaktık. (equal_var parametresine göre kendi seçiyor)


# Adım 2: Elde ettiğiniz test sonuçlarına göre müşteriye tavsiyede bulununuz.

#SO
# Kontrol ve Test grubu arasında "Purchase(satın alma)" bakımından istatistiksel olarak anlamlı bir farklılılık bulunamamıştır,
# müşterimiz İsmail YK reklam teklifleme stratejilerini Click(tıklanma) veya Earning(kazanç) gibi diğer belirleyici faktörlere dayanarak yapmalıdır.

def report(group1, group2, column,sig_level=0.05, Conf=False):
    """
    :param group1: Sample
    :param group2: Sample
    :param column: Feature
    :param sig_level: Significance Level
    :param Conf: Confidence interval
    :return: Report
    """
    group1,group2 = group1.copy(),group2.copy()
    g1_mean, g2_mean = group1[column].mean(), group2[column].mean()
    p_ind, p_mann, p_levene = 0,0,0
    if Conf:
        group1["Group"] = "control"
        group2["Group"] = "test"
        concated = pd.concat([group1, group2], ignore_index=True)
        low, up = sms.DescrStatsW(concated[column]).tconfint_mean()

    w_shapiro1, p_shapiro1 = shapiro(group1[column])
    w_shapiro2, p_shapiro2 = shapiro(group2[column])
    if (p_shapiro1 >= sig_level) & (p_shapiro2 >= sig_level) :
        w_levene, p_levene = levene(group1[column], group2[column])
        t_ind, p_ind, = ttest_ind(group1[column], group2[column], equal_var=(p_levene >= sig_level))
    else:
        t_mann, p_mann = mannwhitneyu(group1[column], group2[column])
    used = "ttest_ind(ortalama)" if p_ind > 0 else "mannwhitneyu(rank)"
    if Conf:
        return (f"{YEL if p_ind > 0 else BLU }"
                f"*Birinci Grup* \nMean-> {g1_mean:.3f}, Normallik-> {(p_shapiro1 >= sig_level)}\n"
                f"*İkinci Grup* \nMean-> {g2_mean:.3f}, Normallik-> {(p_shapiro2 >= sig_level)}\n"
                f"Birinci Grup - İkinci Grup Varyans Benzerliği: {(p_levene >= sig_level)}\n"
                f"Güven Aralığı: {low:.3f} - {up:.3f}\n"
                f"Kullanılan Yöntem: {used}\n"
                f"H₀: {(p_ind or p_mann) >= sig_level} \t H₁: {(p_ind or p_mann) <= sig_level}")
    else:
        return (f"{YEL if p_ind > 0 else BLU }"
                f"*Birinci Grup* \nMean-> {g1_mean:.3f}, Normallik-> {(p_shapiro1 >= sig_level)}\n"
                f"*İkinci Grup* \nMean-> {g2_mean:.3f}, Normallik-> {(p_shapiro2 >= sig_level)}\n"
                f"Birinci Grup - İkinci Grup Varyans Benzerliği: {(p_levene >= sig_level)}\n"
                f"Kullanılan Yöntem: {used}\n"
                f"H₀: {(p_ind or p_mann) >= sig_level} \t H₁: {(p_ind or p_mann) <= sig_level}")

print(report(control,test,column="Purchase",sig_level=0.05, Conf=False))
