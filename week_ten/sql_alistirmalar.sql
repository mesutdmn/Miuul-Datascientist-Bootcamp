1. Customers isimli bir veritabanı ve verilen veri setindeki değişkenleri içerecek FLO isimli bir tablo oluşturunuz.


2. Kaç farklı müşterinin alışveriş yaptığını gösterecek sorguyu yazınız.
SELECT COUNT(master_id) Musteri_Sayısı FROM FLO

3. Toplam yapılan alışveriş sayısı ve ciroyu getirecek sorguyu yazınız.
SELECT COUNT(master_id) Alışveriş_Sayısı, SUM(customer_value_total_ever_offline + customer_value_total_ever_online) Toplam_Ciro FROM FLO

4. Alışveriş başına ortalama ciroyu getirecek sorguyu yazınız.
SELECT COUNT(master_id) Alışveriş_Sayısı, AVG(customer_value_total_ever_offline + customer_value_total_ever_online) Ortalama_Ciro FROM FLO

5. En son alışveriş yapılan kanal (last_order_channel) üzerinden yapılan alışverişlerin toplam ciro ve alışveriş sayılarını getirecek sorguyu yazınız.
#
SELECT last_order_channel, COUNT(master_id) SAYI, SUM(customer_value_total_ever_offline + customer_value_total_ever_online) Ciro
FROM FLO
GROUP BY last_order_channel

6. Store type kırılımında elde edilen toplam ciroyu getiren sorguyu yazınız.
SELECT store_type, SUM(customer_value_total_ever_offline + customer_value_total_ever_online) Ciro
FROM FLO
GROUP BY store_type

7. Yıl kırılımında alışveriş sayılarını getirecek sorguyu yazınız (Yıl olarak müşterinin ilk alışveriş tarihi (first_order_date) yılını
# baz alınız)
SELECT DATEPART(YEAR,first_order_date) YEAR, COUNT(master_id) Alışveris_Sayısı
FROM FLO
GROUP BY DATEPART(YEAR,first_order_date)
ORDER BY YEAR DESC


8. En son alışveriş yapılan kanal kırılımında alışveriş başına ortalama ciroyu hesaplayacak sorguyu yazınız.
SELECT last_order_channel, AVG(customer_value_total_ever_offline + customer_value_total_ever_online) Ortalama_Ciro
FROM FLO
GROUP BY last_order_channel

9. Son 12 ayda en çok ilgi gören kategoriyi getiren sorguyu yazınız.
SELECT interested_in_categories_12, COUNT(interested_in_categories_12) Times
FROM FLO
WHERE last_order_date >= DATEADD(MONTH, -12, last_order_date)
GROUP BY interested_in_categories_12
ORDER BY Times DESC

10. En çok tercih edilen store_type bilgisini getiren sorguyu yazınız.
SELECT store_type, COUNT(store_type) Times
FROM FLO
GROUP BY store_type
ORDER BY Times DESC


11. En son alışveriş yapılan kanal (last_order_channel) bazında, en çok ilgi gören kategoriyi ve bu kategoriden ne kadarlık
 alışveriş yapıldığını getiren sorguyu yazınız.
 SELECT last_order_channel,
interested_in_categories_12,
COUNT(interested_in_categories_12) Times,
SUM(customer_value_total_ever_offline + customer_value_total_ever_online) Ciro
FROM FLO
WHERE last_order_date >= DATEADD(MONTH, -12, last_order_date)
GROUP BY interested_in_categories_12,last_order_channel
ORDER BY last_order_channel,Times DESC

 
12. En çok alışveriş yapan kişinin ID’ sini getiren sorguyu yazınız.
SELECT TOP 1 master_id, SUM(order_num_total_ever_online+order_num_total_ever_offline) Frequency
FROM FLO
GROUP BY master_id
ORDER BY Frequency DESC

13. En çok alışveriş yapan kişinin alışveriş başına ortalama cirosunu ve alışveriş yapma gün ortalamasını (alışveriş sıklığını)
getiren sorguyu yazınız.
SELECT TOP 1 master_id, 
SUM(order_num_total_ever_online+order_num_total_ever_offline) Frequency,
AVG(customer_value_total_ever_offline + customer_value_total_ever_online) Ortalama_Ciro
FROM FLO
GROUP BY master_id
ORDER BY Frequency DESC

14. En çok alışveriş yapan (ciro bazında) ilk 100 kişinin alışveriş yapma gün ortalamasını (alışveriş sıklığını) getiren sorguyu
yazınız.
SELECT TOP 100 master_id, 
SUM(order_num_total_ever_online+order_num_total_ever_offline) Frequency,
SUM(customer_value_total_ever_offline + customer_value_total_ever_online) Ciro
FROM FLO
GROUP BY master_id
ORDER BY Ciro DESC

15. En son alışveriş yapılan kanal (last_order_channel) kırılımında en çok alışveriş yapan müşteriyi getiren sorguyu yazınız.
SELECT last_order_channel, master_id, 
SUM(order_num_total_ever_online+order_num_total_ever_offline) Frequency,
SUM(customer_value_total_ever_offline + customer_value_total_ever_online) Ciro
FROM FLO
GROUP BY last_order_channel, master_id
ORDER BY last_order_channel, Frequency DESC

16. En son alışveriş yapan kişinin ID’ sini getiren sorguyu yazınız. (Max son tarihte birden fazla alışveriş yapan ID bulunmakta.
Bunları da getiriniz.)
SELECT last_order_date, master_id 
FROM FLO
WHERE last_order_date = (SELECT MAX(last_order_date) FROM FLO)
GROUP BY last_order_date, master_id
ORDER BY last_order_date DESC