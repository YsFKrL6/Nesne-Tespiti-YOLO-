# YOLO11s ile PASCAL VOC Veri Seti Üzerinde Çok Sınıflı Nesne Tespiti

## Proje Özeti

Derin Öğrenme dersi kapsamında, açık kaynaklı PASCAL VOC veri setini kullanarak 20 farklı nesne sınıfını görüntüler üzerinde konumlarıyla birlikte tespit eden bir derin öğrenme modeli geliştirildi. YOLO11s mimarisiyle eğitilen modeli doğrulama ve bağımsız test veriler üzerinde değerlendirildi. Ayrıca eğitilen modeli, kullanıcıların fotoğraf yükleyerek nesne tespiti yapabilmesini sağlayan Gradio tabanlı etkileşimli bir web uygulaması yapıldı.

Proje kapsamında veri hazırlama, model eğitimi, performans değerlendirmesi, gerçek dünya görüntüleri üzerinde test ve kullanıcı arayüzü geliştirme aşamaları gerçekleştirildi.

---

## Proje Bilgileri

| Alan | Açıklama |
|---|---|
| Proje başlığı | YOLO11s ile PASCAL VOC Üzerinde Çok Sınıflı Nesne Tespiti |
| Problem türü | Nesne tespiti |
| Veri seti | PASCAL VOC 2007 ve 2012 |
| Toplam görüntü | 21.503 |
| Eğitim / doğrulama / test | 15.052 / 3.225 / 3.226 |
| Model mimarisi | YOLO11s |
| Görüntü boyutu | 640 × 640 piksel |
| Epoch sayısı | 20 |
| Batch size | 32 |
| Optimizasyon algoritması | AdamW |
| Başlangıç öğrenme oranı | 0.001 |
| Eğitim ortamı | Google Colab, NVIDIA Tesla T4 GPU |
| Derin öğrenme kütüphaneleri | PyTorch, Ultralytics |
| Arayüz | Gradio |
| Eğitim süresi | 85,50 dakika |
| Test mAP@0.5 | 0,8157 |
| Test mAP@0.5:0.95 | 0,6274 |
| Test Precision | 0,8248 |
| Test Recall | 0,7373 |
| Tahmin süresi | 6,55 ms/görüntü |


---

## 1. Ödevin Amacı ve Problem Tanımı

Bu projenin amacı, günlük yaşam görüntülerinde bulunan birden fazla nesneyi aynı anda tespit edebilen bir bilgisayarlı görü sistemi geliştirmektir.

Görüntü sınıflandırma, bir görüntünün hangi sınıfa ait olduğunu belirlemeye odaklanırken nesne tespiti, görüntüde bulunan nesnelerin sınıflarını ve konumlarını belirler. Bu nedenle proje kapsamında, aynı görüntüde farklı nesneleri tespit edebilen ve nesnelerin çevresine sınırlayıcı kutular çizebilen nesne tespiti problemi ele alınmıştır.

Projenin temel hedefleri şunlardır:

- PASCAL VOC veri setini eğitim, doğrulama ve test kümelerine ayırmak.
- YOLO11s mimarisiyle çok sınıflı nesne tespiti modeli geliştirmek.
- Model performansını uygun değerlendirme metrikleriyle ölçmek.
- Modelin farklı gerçek dünya koşullarındaki davranışını incelemek.
- Eğitilen modeli kullanıcıların deneyebileceği bir Gradio arayüzüyle sunmak.

## 2. Veri Seti ve Ön İşleme

### 2.1. PASCAL VOC veri seti

Modelin eğitiminde PASCAL VOC 2007 ve PASCAL VOC 2012 veri setleri kullanılmıştır. Veri seti, farklı ortam ve koşullarda çekilmiş görüntülerde nesne sınıflarını ve konumlarını belirlemek amacıyla hazırlanmıştır.

Projede kullanılan 20 sınıf aşağıdaki gruplarda ele alınmıştır:

- **Taşıtlar:** Uçak (`aeroplane`), bisiklet (`bicycle`), tekne (`boat`), otobüs (`bus`), otomobil (`car`), motosiklet (`motorbike`) ve tren (`train`).
- **Canlılar:** Kuş (`bird`), kedi (`cat`), inek (`cow`), köpek (`dog`), at (`horse`), insan (`person`) ve koyun (`sheep`).
- **Günlük yaşam nesneleri:** Şişe (`bottle`), sandalye (`chair`), yemek masası (`diningtable`), saksı bitkisi (`pottedplant`), koltuk (`sofa`) ve televizyon (`tvmonitor`).

### 2.2. Veri kümelerinin oluşturulması

Veri seti, eğitim sürecinde kullanılacak görüntüler ile modelin değerlendirilmesinde kullanılacak görüntüleri birbirinden ayırmak amacıyla üç kümeye bölünmüştür.

| Küme | Görüntü sayısı | Oran |
|---|---:|---:|
| Eğitim (Train) | 15.052 | %70 |
| Doğrulama (Validation) | 3.225 | %15 |
| Test (Test) | 3.226 | %15 |
| **Toplam** | **21.503** | **%100** |

Veri bölme işleminde sabit bir rastgelelik tohumu kullanılarak tekrarlanabilirlik hedeflenmiştir. Görüntü ve etiket eşleşmelerinin korunmasına dikkat edilmiştir.

Test kümesi, eğitim sırasında ağırlık güncellemek veya hiperparametre seçmek için kullanılmamış. Nihai performans, model ve ayarlar belirlendikten sonra bu küme üzerinde ölçülmüştür.

## 3. Model Mimarisi ve Eğitim Ayarları

### 3.1. YOLO11s mimarisinin seçilmesi

Projede tek aşamalı nesne tespiti yaklaşımını kullanan YOLO11s modeli tercih edilmiştir. Tek aşamalı modeller, nesnelerin sınıflarını ve konumlarını aynı tahmin süreci içerisinde belirlemeyi amaçlar.

YOLO11s modelinin seçilmesindeki temel gerekçe, nesne tespiti başarısı ile hesaplama maliyeti arasında dengeli bir seçenek sunmasıdır. Google Colab üzerindeki NVIDIA Tesla T4 GPU kullanılarak modelin eğitilmesi ve değerlendirilmesi hedeflenmiştir.

### 3.2. Eğitim hiperparametreleri

| Parametre | Değer |
|---|---|
| Model | YOLO11s |
| Giriş görüntü boyutu | 640 × 640 |
| Epoch sayısı | 20 |
| Batch size | 32 |
| Başlangıç öğrenme oranı | 0.001 |
| Optimizasyon algoritması | AdamW |
| Early stopping patience | 8 |
| Kayıt periyodu | 5 epoch |
| Donanım | NVIDIA Tesla T4 GPU |

**Epoch sayısı ve erken durdurma:** Eğitim için 20 epoch belirlenmiştir. Doğrulama performansı belirli bir süre boyunca iyileşmezse erken durdurma mekanizmasının eğitimi sonlandırması amaçlanmıştır. `patience=8` ayarı, en iyi doğrulama sonucundan sonra sekiz epoch boyunca iyileşme olmaması durumunda eğitimin durdurulmasını sağlar.

**Batch size ve bellek yönetimi:** Batch size değeri 32 olarak belirlenmiştir. Bu ayar, her ağırlık güncellemesinde kullanılan görüntü sayısını belirler. Gerçek bellek tüketimi görüntü boyutuna, model yapısına ve veri yükleme ayarlarına bağlıdır.

**Optimizasyon:** AdamW, model ağırlıklarını kayıp fonksiyonuna göre güncellemek için kullanılmıştır. Başlangıç öğrenme oranı 0.001 olarak ayarlanmıştır.

### 3.3. Aktivasyon ve kayıp fonksiyonları

YOLO11s mimarisinde kullanılan temel bileşenler şunlardır:

- **SiLU aktivasyonu:** Ağın ilgili katmanlarında doğrusal olmayan özellik dönüşümlerini sağlar.
- **Sigmoid aktivasyonu:** Uygun çıkışlarda bağımsız olasılık tahminleri üretmek için kullanılır.
- **CIoU Loss:** Tahmin kutusu ile gerçek kutu arasındaki örtüşme ve geometrik farklılıkları değerlendirir.
- **DFL (Distribution Focal Loss):** Sınırlayıcı kutu koordinatlarının tahmin edilmesinde kullanılan kayıp bileşenidir.
- **BCE (Binary Cross-Entropy):** İlgili sınıflandırma tahminlerinin hata değerini hesaplamak için kullanılır.

Bu bileşenler birlikte, nesne konumlarının ve sınıflandırma sonuçlarının iyileştirilmesine katkı sağlar.


## 4. Performans Değerlendirmesi

Modelin başarısı, nesne tespiti görevlerinde yaygın olarak kullanılan Precision, Recall, IoU ve mAP metrikleriyle değerlendirilmiştir.

### 4.1. Değerlendirme metrikleri

**IoU (Intersection over Union):** Tahmin edilen sınırlayıcı kutu ile gerçek etiket kutusunun kesişim alanının birleşim alanına oranıdır. Kutuların konumsal örtüşmesini ölçer.

**Precision (Kesinlik):** Modelin doğru olarak değerlendirdiği pozitif tespitlerin, tüm pozitif tespitlere oranını ifade eder. Yanlış pozitif tespitlerin etkisini değerlendirmede kullanılır.

**Recall (Duyarlılık):** Gerçek nesnelerin ne kadarının model tarafından tespit edildiğini gösterir. Kaçırılan nesnelerin değerlendirilmesine yardımcı olur.

**mAP@0.5:** IoU eşiği 0,50 kabul edilerek sınıflar için hesaplanan Average Precision değerlerinin ortalamasıdır.

**mAP@0.5:0.95:** IoU eşiğinin 0,50 ile 0,95 arasında 0,05 aralıklarla değiştirildiği değerlendirmelerdeki ortalama başarıyı ifade eder. Daha katı konumlandırma ölçütlerini de içerdiği için mAP@0.5 değerinden genellikle daha düşüktür.


### 4.2. Bağımsız test kümesi sonuçları

Eğitim ve model seçimi tamamlandıktan sonra en iyi doğrulama performansını veren ağırlıkların bağımsız test kümesi üzerindeki performansı değerlendirilmiştir.

| Metrik | Test sonucu |
|---|---:|
| mAP@0.5 | 0,8157 |
| mAP@0.5:0.95 | 0,6274 |
| Precision | 0,8248 |
| Recall | 0,7373 |
| Eğitim süresi | 85,50 dakika |
| Görüntü başına tahmin süresi | 6,55 ms |

Test sonuçları, modelin farklı nesne sınıflarını tespit edebildiğini göstermektedir. Precision değerinin Recall değerinden yüksek olması, kullanılan değerlendirme ayarlarında doğru pozitif tespitlerin oranının, yakalanan gerçek nesnelerin oranına kıyasla daha yüksek olduğunu göstermektedir.

6,55 ms/görüntü değeri yaklaşık 152 görüntü/saniye hızına karşılık gelir. Ancak bu dönüşüm yalnızca ölçülen süreye dayalı teorik bir hızdır. Gerçek zamanlı performans iddiası için ön işleme, veri aktarımı, son işleme ve donanım koşullarının da hesaba katılması gerekir.


### 4.3. Sınıf bazlı sonuçlar

Bazı sınıflar için aşağıdaki mAP@0.5 değerleri bildirilmiştir:

| Sınıf | Bildirilen mAP@0.5 |
|---|---:|
| Otobüs (`bus`) | 0,935 |
| At (`horse`) | 0,920 |
| Otomobil (`car`) | 0,899 |
| İnsan (`person`) | 0,894 |
| Uçak (`aeroplane`) | 0,880 |
| Kedi (`cat`) | 0,879 |
| Sandalye (`chair`) | 0,624 |
| Saksı bitkisi (`pottedplant`) | 0,663 |
| Tekne (`boat`) | 0,697 |
| Şişe (`bottle`) | 0,701 |

Bu sonuçlar, bazı sınıflarda diğerlerine kıyasla daha yüksek başarı elde edildiğini göstermektedir. Küçük boyutlu, kısmen örtülen veya arka planla görsel açıdan benzerlik gösteren nesnelerde tespit daha zor olabilir.


## 5. Gradio Tabanlı Web Uygulaması

Eğitilen model, kullanıcıların kendi görüntülerini yükleyerek nesne tespiti gerçekleştirebilmeleri için Gradio tabanlı bir web arayüzüne entegre edilmiştir.

Uygulamanın çalışma akışı üç temel adımdan oluşmaktadır:

1. **Görüntü yükleme:** Kullanıcı arayüz üzerinden bir fotoğraf seçer.
2. **Nesne tespiti:** YOLO11s modeli görüntüyü işler ve belirlenen güven eşiğine uygun tespitleri üretir.
3. **Sonuçları gösterme:** Tespit edilen nesneler görüntü üzerinde sınırlayıcı kutularla işaretlenir. Sınıf isimleri ve güven skorları metin alanında gösterilir.

Arayüzdeki güven eşiği 0,10 ile 0,90 arasında ayarlanabilir. Daha düşük eşikler daha fazla tespitin gösterilmesine olanak tanırken yanlış pozitiflerin sayısını artırabilir. Daha yüksek eşikler ise düşük güvenli tespitleri eler ve bazı gerçek nesnelerin sonuçlarda görünmemesine neden olabilir.

Bu arayüz, modelin kullanıcı açısından erişilebilir hâle getirilmesini ve farklı görüntüler üzerinde deneysel olarak test edilmesini sağlamaktadır.

## 6. Gerçek Görüntülerle Test ve Hata Analizi

Modelin veri seti dışındaki gerçek dünya koşullarındaki performansını ve sınırlarını ölçmek amacıyla farklı ışık, yansıma, açı ve kapanma koşullarında çekilen 5 gerçek fotoğraf Gradio web arayüzünde test edilmiştir:

| Fotoğraf | Çekim Koşulu | Gerçekteki Nesneler |
| :--- | :--- |  :--- |
| **1. Durak Camı Yansıması (`01.jpg`)** | Gün ışığı, cam yüzey yansıması ve arka plan örtüşmesi | Yansımadaki insan, arka plandaki otomobiller | 
| **2. Gece Liman ve Tekneler (`02.jpg`)** | Gece düşük ışık, mavi yapay aydınlatma ve bitişik tekneler | Tekneler | 
| **3. Sahil Masa Düzeni (`03.jpg`)** | ışık, ağaç gövdesi arkası kapanma | Masa, Sandalyeler |
| **4. Havalimanı  (`04.jpg`)** | Gün ışığı, yakın plan kanat ve uzak mesafe nesneler | Uçaklar | 
| **5. Gece Motosiklet ve Bitki (`05.jpg`)** | Gece yapay sokak aydınlatması, ön tekerde trafik dubası engeli | Motosiklet, Küçük bitki |

### Başarısız Durumların Değerlendirmesi

- **Hata Yaptığı Durumlar:**
  - **Kapanma ve Düşük Kontrast (`03.jpg`):** Sahil düzeninde masa ve dört sandalye tespit edilmesine rağmen, en sağdaki sandalyede oturan kişi ağaç gövdesi ve yaprakların oluşturduğu kısmi kapanma ile düşük arka plan kontrastı nedeniyle tespit edilememiştir.
  - **Düşük Işıkta Kutu Birleşmesi (`02.jpg`):** Gece liman görüntüsünde yan yana demirli iki tekne olarak doğru sınıflandırılmış, ancak düşük ışık ve koyu gövde sınırları nedeniyle iki ayrı nesne yerine tek bir sınırlayıcı kutu içinde birleştirilmiştir.
  - **Ölçek ve Kısmi Görünüm Farkları (`04.jpg`):** Aprondaki yolcu uçağı başarıyla tespit edilirken, kadrajın solundaki küçük ölçekli yer hizmet aracı eşik değerinin altında kalmış ve ön planda yalnızca kanat kısmı görünen yakın uçak bütüncül silüet eksikliği nedeniyle etiketlenmemiştir.
  - **Sınıf Benzerliği (`05.jpg`):** Sol tarafta doğrudan toprağa dikili küçük palmiye bitkisi, görsel form benzerliği nedeniyle saksı bitkisi sınıfına atanmıştır.

---


## 7. GitHub Depo İçeriği

Proje deposunda aşağıdaki dosyalar yer almaktadır:

- **`YOLO11_Object_Detection.ipynb`:** Veri hazırlama, eğitim, değerlendirme ve Gradio arayüzü için kullanılan Jupyter Notebook.
- **`voc_custom.yaml`:** Veri seti yolları ve 20 sınıfa ilişkin yapılandırma.
- **`best.pt`:** Eğitim sırasında en iyi doğrulama performansını veren model ağırlıkları.
- **`results.png`:** Eğitim ve doğrulama metriklerinin grafiksel gösterimi.
- **`confusion_matrix.png`:** Sınıflandırma hatalarının sınıflara göre dağılımı.
- **`BoxPR_curve.png`:** Precision-Recall eğrisi.
- **`BoxF1_curve.png`:** F1 skorunun güven eşiğine göre değişimi.
- **Örnek test çıktıları:** Modelin farklı görüntülerde ürettiği tespit sonuçları.
  

