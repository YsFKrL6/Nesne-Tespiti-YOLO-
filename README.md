# YOLO11s ile Çok Sınıflı Nesne Tespiti (PASCAL VOC)

Derin öğrenme dersi ödevi kapsamında, açık kaynaklı PASCAL VOC veri setini kullanarak 20 farklı nesne sınıfını konumlarıyla birlikte tespit eden bir derin öğrenme modeli geliştirdim. Eğittiğim modeli eğitim sürecinde hiç görmediği bağımsız test verileriyle değerlendirdikten sonra, kullanıcıların kendi fotoğraflarını yükleyip anlık tespit alabilmesi için **Gradio** tabanlı etkileşimli bir web uygulamasına dönüştürdüm.

---

## Proje Raporu Özeti

| Alan | Açıklama |
| :--- | :--- |
| **Proje Başlığı** | YOLO11s ile PASCAL VOC Veri Seti Üzerinde Çok Sınıflı Nesne Tespiti |
| **Problem Türü** | Nesne Tespiti (Object Detection — 20 Sınıf) |
| **Veri Seti** | PASCAL VOC 2007 + 2012 Birleşik Veri Seti (Toplam 21.503 görüntü: 15.052 Eğitim, 3.225 Doğrulama, 3.226 Test) |
| **Veri Seti Kaynağı** | http://host.robots.ox.ac.uk/pascal/VOC/ |
| **Model Mimarisi** | YOLO11 |
| **Giriş Görüntü Boyutu** | 640 × 640 piksel |
| **Hiperparametreler** | 20 Epoch, Batch Size: 32, Başlangıç Öğrenme Oranı: 0.001, Early Stopping: 8 Epoch, Kayıt Periyodu: 5 Epoch |
| **Optimizasyon Algoritması** | AdamW |
| **Aktivasyon Fonksiyonları** | Gizli katmanlarda **SiLU**, sınıf olasılık çıkışlarında **Sigmoid** |
| **Kayıp (Loss) Fonksiyonları** | **CIoU Loss** + **DFL** (kutu konumlandırma) ve **BCE Loss** (sınıflandırma) |
| **Değerlendirme Metrikleri** | IoU, Precision, Recall, mAP@0.5, mAP@0.5:0.95, Eğitim Süresi, Tahmin Süresi (ms) |
| **Eğitim Ortamı** | Google Colab — NVIDIA Tesla T4 GPU, PyTorch & Ultralytics |
| **En İyi Test Performansı** | Test mAP@0.5: **0.8157**, mAP@0.5:0.95: **0.6274**, Precision: **0.8248**, Recall: **0.7373** *(Eğitim Süresi: 85.50 dk)* |
| **Tahmin Hızı (Inference)** | **6.55 ms** / görüntü |
| **Uygulama** | Gradio Web Arayüzü |

---

## 1. Ödevin Amacı ve Seçilen Görev

Bu çalışmada, günlük yaşam sahnelerindeki birden fazla nesneyi hem sınıflandırabilen hem de görüntü üzerindeki koordinatlarını sınırlayıcı kutularla işaretleyebilen uçtan uca bir bilgisayarlı görü sistemi geliştirdim. Görüntü sınıflandırma yalnızca tek bir baskın etiketi verirken, **Nesne Tespiti (Object Detection)** yaklaşımı tek bir karedeki farklı boyut, açı ve konumlardaki çok sayıda nesneyi aynı anda tespit edebildiği için problem türü olarak bu görevi seçtim.

---

## 2. Veri Seti ve Ön İşleme Süreci

Model eğitiminde nesne tespiti literatüründe standart ölçüt kabul edilen **PASCAL VOC 2007 ve 2012** birleşik veri setini kullandım. Veri seti toplam **21.503 görüntü** ve günlük hayattan **20 farklı nesne sınıfı** içermektedir:
* **Taşıtlar:** Uçak (`aeroplane`), bisiklet (`bicycle`), tekne (`boat`), otobüs (`bus`), otomobil (`car`), motosiklet (`motorbike`), tren (`train`)
* **Canlılar:** Kuş (`bird`), kedi (`cat`), inek (`cow`), köpek (`dog`), at (`horse`), insan (`person`), koyun (`sheep`)
* **İç Mekân ve Günlük Nesneler:** Şişe (`bottle`), sandalye (`chair`), yemek masası (`diningtable`), saksı bitkisi (`pottedplant`), koltuk (`sofa`), televizyon (`tvmonitor`)

### Ön İşleme ve Veri Sızıntısını (Data Leakage) Önleme Stratejisi
* **Bağımsız Bölme Yapısı:** Aynı görüntünün eğitim ve test kümelerine birlikte düşerek yapay bir başarı skoru oluşturmasını (data leakage) önlemek amacıyla tüm görüntü-etiket çiftleri sabit bir rastgelelik tohumu (seed) ile karıştırılarak üç bağımsız kümeye ayrılmıştır:
  * **Eğitim Kümesi (Train — %70):** 15.052 görüntü
  * **Doğrulama Kümesi (Validation — %15):** 3.225 görüntü
  * **Test Kümesi (Test — %15):** 3.226 görüntü

---

## 3. Model Mimarisi ve Hiperparametre Gerekçeleri

Mimari olarak tek aşamalı (one-stage) çalışan güncel **YOLO11s** modelini tercih ettim. `Small` (`s`) ölçeğini seçmemin temel nedeni, Google Colab T4 GPU donanımında makul bir sürede eğitilebilmesi ve gerçek zamanlı çıkarım hızında yüksek doğruluk sunmasıdır.

### Seçilen Ayarlar ve Teknik Gerekçeleri
* **Epoch Sayısı (20) ve Erken Durdurma (Patience = 8):** 15.052 eğitim görüntüsünde ağırlıkların yeterince yakınsayabilmesi için 20 epoch belirledim. Doğrulama başarısı art arda 8 epoch boyunca gelişmezse aşırı öğrenmeyi (overfitting) önlemek amacıyla eğitimi otomatik durduracak erken durdurma mekanizmasını aktif ettim.
* **Yığın Boyutu (Batch Size = 32) ve Bellek Yönetimi:** T4 GPU'nun 15 GB VRAM kapasitesini verimli kullanmak için yığın boyutunu 32 seçtim. Google Colab oturumunun 12.7 GB sistem RAM sınırını aşarak oturumun çökmesini engellemek için görüntüleri RAM'e önbellekleme özelliğini kapalı tuttum.
* **Optimizasyon ve Öğrenme Oranı (AdamW, lr0 = 0.001):** Ön eğitimli temel görsel özellikleri bozmadan PASCAL VOC sınıflarına hassas ince ayar (fine-tuning) yapabilmek için ağırlık çürümesi destekli **AdamW** optimizasyonunu ve `0.001` başlangıç öğrenme oranını kullandım.
* **Aktivasyon ve Kayıp Fonksiyonları:** Ağın gizli katmanlarında gradyan akışını koruyan **SiLU**, çıkış katmanında ise çoklu sınıf olasılıklarını bağımsız hesaplayan **Sigmoid** aktivasyon fonksiyonu kullanılmıştır. Kayıp fonksiyonu olarak kutu konumlandırma örtüşmesini iyileştiren **CIoU + DFL** ve sınıf ayrımını ölçen **BCE (Binary Cross-Entropy)** birlikte çalışmaktadır.

---

## 4. Performans Değerlendirmesi ve Metrik Analizi

### Kullanılan Metriklerin Anlamı
* **IoU (Intersection over Union):** Modelin çizdiği tahmin kutusu ile gerçek etiket kutusunun kesişim alanının birleşim alanına oranıdır.
* **Precision (Kesinlik):** Modelin "nesne var" diyerek işaretlediği kutuların yüzde kaçının gerçekten doğru olduğunu gösterir.
* **Recall (Duyarlılık):** Görüntüdeki gerçek nesnelerin yüzde kaçının model tarafından kaçırılmadan yakalandığını ölçer.
* **mAP@0.5:** En az %50 IoU örtüşme eşiğinde 20 sınıfın ortalama doğruluk (Average Precision) değeridir.
* **mAP@0.5:0.95:** IoU eşiği %50'den %95'e kadar 0.05 adımlarla artırılırken hesaplanan daha katı ve hassas ortalama başarı skorudur.

### Eğitim Süreci Doğrulama (Validation) Gelişimi
20 epoch boyunca 3.225 doğrulama görüntüsü üzerinde kaydedilen ilerleme, modelin düzenli olarak öğrendiğini göstermektedir:

| Epoch | Precision (P) | Recall (R) | mAP@0.5 | mAP@0.5:0.95 |
| :--- | :---: | :---: | :---: | :---: |
| **1 / 20** | 0.392 | 0.192 | 0.176 | 0.0932 |
| **4 / 20** | 0.548 | 0.499 | 0.525 | 0.3420 |
| **8 / 20** | 0.714 | 0.593 | 0.674 | 0.4710 |
| **10 / 20** | 0.704 | 0.647 | 0.704 | 0.5050 |
| **15 / 20** | 0.773 | 0.720 | 0.787 | 0.5910 |
| **20 / 20 (Final)** | **0.821** | **0.735** | **0.815** | **0.6290** |

### Bağımsız Test Kümesi Sonuçları
Eğitim tamamlandıktan sonra en yüksek doğrulama skorunu veren model ağırlıkları yüklenerek, eğitimde hiç görülmemiş **3.226 adet test görüntüsü** (toplam 7.752 nesne örneği) değerlendirilmiştir:

* **Toplam Eğitim Süresi:** `85.50` dakika (`1.407` saat)
* **Test mAP@0.5:** `0.8157`
* **Test mAP@0.5:0.95:** `0.6274`
* **Test Precision:** `0.8248`
* **Test Recall:** `0.7373`
* **Tahmin Süresi (Inference Time):** `6.55 ms` / görüntü (~152 FPS)

### Bulguların Teknik Yorumu
* **Genelleme Başarısı:** Doğrulama kümesinde `0.8150` olan mAP@0.5 skorunun, tamamen bağımsız test kümesinde `0.8157` olarak gerçekleşmesi modelde **aşırı öğrenme (overfitting) yaşanmadığını** ve genelleme kapasitesinin yüksek olduğunu kanıtlamaktadır.
* **Sınıf Bazlı Başarı Farkları:** Belirgin hatlara ve büyük ölçeğe sahip sınıflarda (`bus`: 0.935, `horse`: 0.920, `car`: 0.899, `person`: 0.894, `aeroplane`: 0.880, `cat`: 0.879) oldukça yüksek doğruluk elde edilmiştir. Buna karşın küçük boyutlu, arka planla iç içe geçen ve birbirini kısmen kapatan nesnelerde (`chair`: 0.624, `pottedplant`: 0.663, `boat`: 0.697, `bottle`: 0.701) başarının görece düştüğü gözlemlenmiştir.

---

## 5. Uygulama (Deployment) — Gradio Web Arayüzü

Eğitilen modeli son kullanıcıların kolayca test edebileceği etkileşimli bir web uygulamasına dönüştürmek için **Gradio** arayüzü tasarladım. Uygulamanın çalışma akışı şu adımlardan oluşur:

1. **Görüntü Yükleme:** Kullanıcı web arayüzü üzerinden kendi cihazındaki herhangi bir fotoğrafı sisteme yükler.
2. **Dinamik Güven Eşiği (Confidence Threshold):** Kullanıcı arayüzdeki kaydırıcı (slider) yardımıyla `0.10` ile `0.90` arasında bir güven eşiği seçebilir (varsayılan değer `0.40`). Eşik düşürüldüğünde daha fazla aday nesne yakalanırken yanlış pozitif riski artar; eşik yükseltildiğinde yalnızca yüksek kesinlikli tahminler gösterilir.
3. **Görsel ve Metinsel Çıktı:** Model görüntüyü 640×640 boyutunda işleyerek tespit ettiği nesnelerin etrafına sınırlayıcı kutular çizer, renk uzayı dönüşümünü yaparak işlenmiş görüntüyü ekrana yansıtır ve sağ panelde tespit edilen sınıfları yüzdelik güven skorlarıyla birlikte listeler.

---

## 6. Gerçek Görüntülerle Test ve Hata Analizi

Modelin veri seti dışındaki gerçek dünya koşullarındaki performansını ve sınırlarını ölçmek amacıyla farklı ışık, yansıma, açı ve kapanma (occlusion) koşullarında kendi çektiğim fotoğrafları Gradio web arayüzünde test ettim:

| Fotoğraf | Çekim Koşulu | Gerçekteki Nesneler | Modelin Tahmini ve Gözlem | Durum |
| :--- | :--- | :--- | :--- | :--- |
| **1. Durak Camı Yansıması (`01.jpg`)** | Gün ışığı, cam yüzey yansıması ve arka plan örtüşmesi | İnsan (`person`), Otomobiller (`car`) | Yansımadaki kişi ve arka plandaki araçlar tespit edildi; cam yansıması nedeniyle güven skorları görece düşük kaldı | Kısmen Başarılı |
| **2. Gündüz Trafik (`13.jpg`)** | Gün ışığı, çoklu araç | Otomobiller (`car`) | Ön ve orta plandaki otomobiller yüksek güven skorlarıyla doğru işaretlendi | Başarılı |
| **3. Sahil ve Vapur (`03.jpg`)** | Gün ışığı, çoklu insan ve deniz aracı | İnsanlar (`person`), Vapur (`boat`) | Sahil hattındaki yayalar ve deniz üzerindeki vapur doğru sınıflandırıldı | Başarılı |
| **4. Havalimanı Apronu (`06.jpg`)** | Gün ışığı, uzak uçak ve yakın kanat | Uçak (`aeroplane`), Araç (`car`) | Aprondaki uçak ve yer hizmet aracı tespit edildi; aşırı uzak nesneler eşik altında kaldı | Başarılı / Kısmi |
| **5. Sahil Masa Düzeni (`10.jpg`)** | Alacakaranlık, ağaç arkası kapanma | Masa (`diningtable`), Sandalye (`chair`), İnsan (`person`) | Açıkta kalan masa ve sandalyeler yakalandı; ağaç gövdesi arkasında kalan kişiler kısmen kaçırıldı | Kısmen Başarılı |
| **6. Gece Motosiklet (`08.jpg`)** | Gece yapay ışık, ön tekerde duba engeli | Motosiklet (`motorbike`) | Yapay sokak aydınlatması altında motosiklet yüksek doğrulukla tespit edildi | Başarılı |
| **7. Asfalt Üzerinde Karga (`17.jpg`)** | Tepeden bakış açısı, düşük kontrastlı gri zemin | Kuş (`bird`) | Varsayılan `0.40` eşiğinde tespit yok; eşik düşürüldüğünde `sheep` (Yanlış Pozitif) olarak etiketlendi | Başarısız (Hata Analizi) |
| **8. Gece Meydan (`gece_meydan.jpg`)** | Gece ışığı, hareket bulanıklığı ve kalabalık | İnsanlar (`person`), Polis Aracı (`car`) | Sabit duran kişiler ve araç tespit edildi; hareket bulanıklığı olan uzak hedeflerde güven skoru düştü | Zorlandı / Kısmi |

### Başarılı ve Başarısız Durumların Değerlendirmesi
* **Başarılı Olduğu Durumlar:** Gün ışığında çekilen, arka plandan net biçimde ayrışan ve kadraja tam giren otomobil, uçak, motosiklet ve yürüyen insan örneklerinde model yüksek güven skorlarıyla doğru sınırlayıcı kutular çizmiştir.
* **Zorlandığı / Başarısız Olduğu Durumlar:**
  * **Cam Yansıması ve Yarı Saydamlık (`01.jpg`):** Durak camındaki yansıma görüntüsü ile camın arkasından geçen araçların aynı piksel bölgesinde çakışması kontrast kaybına yol açarak güven skorlarını düşürmüştür.
  * **Tepeden Açı ve Doku Benzerliği (`17.jpg`):** Asfalt zemin üzerindeki karga fotoğrafında varsayılan `0.40` eşiğinde nesne yakalanamamış; eşik düşürüldüğünde ise üstten çekim silüeti ve gri tüy-asfalt dokusu benzerliği nedeniyle model nesneyi yanlış pozitif (false positive) olarak `sheep` (koyun) sınıfına atamıştır.
  * **Düşük Işık, Kapanma ve Hareket Bulanıklığı (`10.jpg`, `gece_meydan.jpg`):** Alacakaranlıkta ağaç gövdesi arkasında kısmen gizlenen kişilerde (occlusion) ve gece hareket bulanıklığı (motion blur) oluşan karelerde tespit doğruluğunun azaldığı gözlemlenmiştir.

---

## 7. Depo (Repository) İçeriği

* **Kod Defteri (`YOLO11_Object_Detection.ipynb`):** Veri hazırlama, model eğitimi, bağımsız test değerlendirmesi ve Gradio web arayüzünün tüm kaynak kodları ile çalışma çıktıları.
* **Yapılandırma Dosyası (`voc_custom.yaml`):** Veri seti dizin yolları ve 20 nesne sınıfının tanımı.
* **Eğitilmiş Model Ağırlıkları (`best.pt`):** Eğitim boyunca en yüksek doğrulama skorunu elde eden nihai model dosyası (19.2 MB).
* **Performans Grafikleri (`results.png`, `confusion_matrix.png`, `BoxPR_curve.png`, `BoxF1_curve.png`):** Eğitim/doğrulama kayıp eğrileri, 20 sınıf karmaşıklık matrisi, Precision-Recall ve F1 eğrileri.
* **Gerçek Dünya Test Çıktıları:** Gradio web arayüzü üzerinden test edilen gerçek fotoğrafların tespit görselleri.
