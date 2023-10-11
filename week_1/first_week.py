# conda env list : mevcut environmentları listeler. içinde bulunduğun sanal ortam yıldız ifadesi alır.
# conda create -n myenv : myenv adında bir sanal ortam olusturmak için kullanılır
# conda activate myenv : yeni olusturdugumuz sanal ortamı aktif etmek için kullanılır.
# conda deactivate : base ortamına geri dönmemizi sağlar
# conda list : ortam içerisindeki paketleri listeler
# conda install numpy : numpynin son versiyonunu sanal ortamımıza yükler
# conda install numpy scipy pandas : aynı anda birkaçtane birden
# conda remove package_name : yüklü bir paketi silmek için
# conda install numpy=1.20.1 : condada = , pipte ==
# conda upgrade numpy : çalışma ortamımızdaki numpyi günceller.
# conda upgrade -all : tüm kütüphaneleri günceller
# pip install paket_adi==version : pipin kullanım şekli
# conda env export > environment.yaml : kullanılan kütüphanelerin listesini dışarı aktarır
# conda env -remove -n myenv : oluşturduğumuz env silmek için kullanılır, önce base dön
# conda env create -f environment.yaml : yaml dosyasındaki bilgileri kullanrak bir sanal ortam oluşturmaya yarar
