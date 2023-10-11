###############################################
# Python Alıştırmalar
###############################################

###############################################
# GÖREV 1: Veri yapılarının tipleriniz inceleyiniz.
###############################################

x = 8
type(x)
y = 3.2
type(y) #float
z = 8j + 18
type(z) #complex

a = "Hello World"
type(a) #String

b = True
type(b) #bool

c = 23 < 22
type(c) #Bool


l = [1, 2, 3, 4,"String",3.2, False]
type(l) #list


d = {"Name": "Jake",
     "Age": [27,56],
     "Adress": "Downtown"}
type(d) #dict

t = ("Machine Learning", "Data Science")

type(t) #tuple

s = {"Python", "Machine Learning", "Data Science","Python"}

type(s) #set


###############################################
# GÖREV 2: Verilen string ifadenin tüm harflerini büyük harfe çeviriniz.
# Virgül ve nokta yerine space koyunuz, kelime kelime ayırınız.
###############################################

text = "The goal is to turn data into information, and information into insight."
text.upper().replace(","," ").replace("."," ").split()

#['THE', 'GOAL', 'IS', 'TO', 'TURN', 'DATA', 'INTO', 'INFORMATION', 'AND', 'INFORMATION', 'INTO', 'INSIGHT']


###############################################
# GÖREV 3: Verilen liste için aşağıdaki görevleri yapınız.
###############################################

lst = ["D","A","T","A","S","C","I","E","N","C","E"]
a = [lst[0],lst[8]]

# Adım 1: Verilen listenin eleman sayısına bakın.

len(lst)
#11

# Adım 2: Sıfırıncı ve onuncu index'teki elemanları çağırın.

lst[0]
# D
lst[10]
# E

# Adım 3: Verilen liste üzerinden ["D","A","T","A"] listesi oluşturun.

lst[0:4]

#['D', 'A', 'T', 'A']

# Adım 4: Sekizinci index'teki elemanı silin.
lst.pop(8)
lst
#['D', 'A', 'T', 'A', 'S', 'C', 'I', 'E', 'C', 'E']

# Adım 5: Yeni bir eleman ekleyin.
lst.append("M")
lst
#['D', 'A', 'T', 'A', 'S', 'C', 'I', 'E', 'C', 'E', 'M']

# Adım 6: Sekizinci index'e  "N" elemanını tekrar ekleyin.
lst.insert(8,"N")
lst
#['D', 'A', 'T', 'A', 'S', 'C', 'I', 'E', 'N', 'C', 'E', 'M']


###############################################
# GÖREV 4: Verilen sözlük yapısına aşağıdaki adımları uygulayınız.
###############################################

dict = {'Christian': ["America",18],
        'Daisy':["England",12],
        'Antonio':["Spain",22],
        'Dante':["Italy",25]}


# Adım 1: Key değerlerine erişiniz.
dict.keys()
#dict_keys(['Christian', 'Daisy', 'Antonio', 'Dante'])


# Adım 2: Value'lara erişiniz.

dict.values()
#dict_values([['America', 18], ['England', 12], ['Spain', 22], ['Italy', 25]])

# Adım 3: Daisy key'ine ait 12 değerini 13 olarak güncelleyiniz.

dict["Daisy"][1] = 13

#{'Christian': ['America', 18], 'Daisy': ['England', 13], 'Antonio': ['Spain', 22], 'Dante': ['Italy', 25]}
import timeit
# Adım 4: Key değeri Ahmet value değeri [Turkey,24] olan yeni bir değer ekleyiniz.

%timeit -r3 -n1000 dict["Ahmet"] = ["Turkey",24]
%timeit -r3 -n1000 dict.update({"Ahmet": ["Turkey",24]})
dict

# {'Christian': ['America', 18], 'Daisy': ['England', 13], 'Antonio': ['Spain', 22], 'Dante': ['Italy', 25], 'Ahmet': ['Turkey', 24]}


# Adım 5: Antonio'yu dictionary'den siliniz.
# {'Christian': ['America', 18], 'Daisy': ['England', 13], 'Dante': ['Italy', 25], 'Ahmet': ['Turkey', 24]}
dict.pop("Antonio")
del dict["Antonio"]
dict.__delitem__("Antonio")

###############################################
# GÖREV 5: Arguman olarak bir liste alan,
# listenin içerisindeki tek ve çift sayıları ayrı listelere atıyan ve bu listeleri return eden fonskiyon yazınız.
###############################################

l = [2,13,18,93,22]

%timeit -r3 -n1000 l[slice(0,3)]
%timeit -r3 -n1000 l[0:3]

def even_or_odd(liste):
    A,B = [],[]
    for i in liste:
        if i % 2 == 0:
            A.append(i)
        else:
            B.append(i)
    return A,B

print(f"Çiftler = {ciftler}") # Çiftler = [2, 18, 22]
print(f"Tekler = {tekler}") # Tekler = [13, 93]

l = [2,13,18,93,22]
def cift_tek(liste):
    return [num for num in liste if num %2 == 0],[num for num in liste if num %2 != 0]

print("Çift Sayılar: ",cift)
print("Tek Sayılar: ",tek)

%timeit -r3 -n1000 ciftler, tekler = even_or_odd(l)
%timeit -r3 -n1000 cift,tek = cift_tek(l)

###############################################
# GÖREV 6: Aşağıda verilen listede mühendislik ve tıp fakülterinde dereceye giren öğrencilerin isimleri bulunmaktadır.
# Sırasıyla ilk üç öğrenci mühendislik fakültesinin başarı sırasını temsil ederken son üç öğrenci de
# tıp fakültesi öğrenci sırasına aittir.
# Enumarate kullanarak öğrenci derecelerini fakülte özelinde yazdırınız.
###############################################

ogrenciler = ["Ali","Veli","Ayşe","Talat","Zeynep","Ece"]

for index, student in enumerate(ogrenciler, 1):
    if index < 4:
        print(f"Mühendislik Fakültesi {index}. öğrenci: {student}")
    else:
        print(f"Tıp Fakültesi {index-3}. öğrenci: {student}")


###############################################
# GÖREV 7: Aşağıda 3 adet liste verilmiştir.
# Listelerde sırası ile bir dersin kodu, kredisi ve kontenjan bilgileri yer almaktadır.
# Zip kullanarak ders bilgilerini bastırınız.
###############################################

ders_kodu = ["CMP1005","PSY1001","HUK1005","SEN2204"]
kredi = [3,4,2,4]
kontenjan = [30,75,150,25]


for ders, krd, kon in zip(ders_kodu, kredi, kontenjan):
    print(f"Kredisi {krd} olan {ders} kodlu dersin kontenjanı {kon} kişidir.")


###############################################
# GÖREV 8: Aşağıda 2 adet set verilmiştir.
# Sizden istenilen eğer 1. küme 2. kümeyi kapsiyor ise ortak elemanlarını
# eğer kapsamıyor ise 2. kümenin 1. kümeden farkını yazdıracak fonksiyonu tanımlamanız beklenmektedir.
###############################################

kume1 = set([ "python","data"])
kume2 = set(["data", "function", "qcut", "lambda", "python", "miuul"])

def set_check(set_1, set_2):

    if set_1.issuperset(set_2):
        print(f"Küme 1 ve Küme 2 Ortak Elemanlar: {set_1 & set_2}")
    else:
        print(f"Küme 2'nin Küme 1 den Farkı : {set_2 - set_1}")

%timeit -r3 -n10000 kume2.difference(kume1)
%timeit -r3 -n10000 kume2 - kume1

def alternating(string):
    new_string = ""

    for string_index in range(len(string)):
        if string_index %2 == 0:
            new_string += string[string_index].upper()
        else:
            new_string += string[string_index].lower()
    return new_string

alternating("miuul")

def alternating_1(string):
    new_string = ""

    for i,s in enumerate(string):
        if i %2 == 0:
            new_string += string[i].upper()
        else:
            new_string += string[i].lower()
    return new_string

alternating_1("miuul")

%timeit -r3 -n10000 alternating("miuul")
%timeit -r3 -n10000 alternating_1("miuul")
