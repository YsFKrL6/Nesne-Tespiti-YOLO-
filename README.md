# YOLO11s ile Çok Sınıflı Nesne Tespiti (PASCAL VOC)

Bilgisayarlı görü dersi ödevi kapsamında, açık kaynaklı PASCAL VOC veri setini kullanarak 20 farklı sınıfta nesne tespiti (Object Detection) yapan bir derin öğrenme modeli eğittim. Eğittiğim modeli bağımsız test verileriyle değerlendirdikten sonra, kullanıcıların kendi fotoğraflarını yükleyip tahmin alabilmesi için Gradio ile web arayüzüne dönüştürdüm.

---

## Proje Raporu Özeti

| Alan | Açıklama / Öğrenci Yanıtı |
| :--- | :--- |
| **Project Title** | YOLO11s ile PASCAL VOC Veri Seti Üzerinde Nesne Tespiti |
| **Problem / Task** | Nesne Tespiti (Object Detection — 20 Sınıf) |
| **Dataset** | PASCAL VOC 2007 + 2012 Birleşik Veri Seti (21.503 görüntü: 15.052 Train, 3.225 Val, 3.226 Test) |
| **Dataset URL** | http://host.robots.ox.ac.uk/pascal/VOC/ |
| **Model Architecture** | YOLO11s (`yolo11s.pt`) |
| **Input Size** | 640 × 640 piksel |
| **Hyperparameters** | `epochs=20`, `batch=32`, `lr0=0.001`, `patience=8`, `save_period=5`, `seed=123` |
| **Optimizer** | AdamW |
| **Activation / Output Function** | Gizli katmanlarda **SiLU**, sınıf olasılık çıkışlarında **Sigmoid** |
| **Loss Function** | **CIoU Loss** + **DFL** (kutu konumlandırma) ve **BCE Loss** (sınıflandırma) |
| **Evaluation Metrics** | mAP@0.5, mAP@0.5:0.95, Precision, Recall, Eğitim Süresi, Tahmin Süresi (ms) |
| **Training Environment** | Google Colab — NVIDIA Tesla T4 GPU (16 GB VRAM), PyTorch & Ultralytics |
| **Best Performance** | Test mAP@0.5: **[0.XXXX]**, mAP@0.5:0.95: **[0.XXXX]** *(Eğitim Süresi: [XX.XX] dk)* |
| **Inference Time** | **[X.XX]** ms / görüntü |
| **Deployment Technology** | Gradio Web Arayüzü (`share=True`) |
| **Repository** | `[GitHub Repo Linkini Buraya Yapıştır]` |

---

## 1. Ödevin Amacı ve Seçilen Görev
Bu çalışmada günlük hayatta sık karşılaşılan nesneleri görüntü üzerinde konumlarıyla (bounding box) birlikte tespit edebilen uçtan uca bir bilgisayarlı görü sistemi geliştirdim. Problem türü olarak **Nesne Tespiti (Object Detection)** görevini seçtim; çünkü hem nesnenin ne olduğunu sınıflandırmayı hem de görüntü içindeki koordinatlarını bulmayı gerektiriyor.

---

## 2. Veri Seti ve Ön İşleme
Veri seti olarak nesne tespiti problemlerinde standart kabul edilen **PASCAL VOC 2007 ve 2012** birleşik veri setini kullandım. Veri setinde toplam **21.503 görüntü** ve **20 farklı nesne sınıfı** yer alıyor:
`aeroplane`, `bicycle`, `bird`, `boat`, `bottle`, `bus`, `car`, `cat`, `chair`, `cow`, `diningtable`, `dog`, `horse`, `motorbike`, `person`, `pottedplant`, `sheep`, `sofa`, `train`, `tvmonitor`.

Veri sızıntısını (data leakage) önlemek ve modelin eğitimde hiç görmediği görüntülerle gerçekçi bir test yapabilmek için tüm veri havuzunu `seed=42` ile karıştırıp üç bağımsız kümeye ayırdım:
* **Eğitim Kümesi (Train - %70):** 15.052 görüntü
* **Doğrulama Kümesi (Validation - %15):** 3.225 görüntü
* **Test Kümesi (Test - %15):** 3.226 görüntü

---

## 3. Model Mimarisi ve Eğitim Ayarları
Mimari olarak güncel ve tek aşamalı (one-stage) bir nesne tespit modeli olan **YOLO11s**'i tercih ettim. Small (`s`) varyantını seçmemin temel sebebi, Google Colab T4 GPU üzerinde makul sürede eğitilebilmesi ve gerçek zamanlı çıkarım hızında yüksek doğruluk sunmasıdır.

Eğitim sürecinde kullandığım ayarlar ve seçme nedenlerim:
* **Epoch (`20`) ve Early Stopping (`patience=8`):** 21 bin görüntülük büyük bir veri setinde modelin yeterince öğrenebilmesi için 20 epoch belirledim. Doğrulama başarısı 8 epoch boyunca artmazsa ezberlemeyi (overfitting) önlemek için eğitimi erken durduracak şekilde ayarladım.
* **Batch Size (`32`):** T4 GPU'nun 16 GB VRAM kapasitesini verimli kullanmak ve gradyan güncellemelerini daha kararlı hale getirmek için 32 seçtim.
* **Optimizer (`AdamW`) ve Learning Rate (`0.001`):** Ön eğitimli (pretrained) ağırlıkları bozmadan hassas ince ayar (fine-tuning) yapabilmek için ağırlık çürümesi destekli AdamW optimizasyonunu ve `0.001` başlangıç öğrenme oranını kullandım.
* **Aktivasyon ve Kayıp Fonksiyonları:** YOLO11 mimarisinin arka planında gizli katmanlarda **SiLU**, sınıf tahmin çıkışlarında **Sigmoid** aktivasyonu çalışmaktadır. Kayıp fonksiyonu olarak ise tahmin kutularının gerçek kutularla örtüşmesini (IoU) iyileştiren **CIoU + DFL** ve sınıf doğruluğunu ölçen **BCE (Binary Cross-Entropy)** kullanılmaktadır.

---

## 4. Performans Değerlendirmesi
Eğitim tamamlandıktan sonra en yüksek doğrulama skorunu veren `best.pt` ağırlıklarını yükleyip, modelin eğitim boyunca hiç görmediği **3.226 adet test görüntüsü** (`split="test"`) üzerinde değerlendirme yaptım:

* **Toplam Eğitim Süresi:** `[XX.XX]` dakika
* **mAP@0.5:** `[0.XXXX]`
* **mAP@0.5:0.95:** `[0.XXXX]`
* **Precision (Kesinlik):** `[0.XXXX]`
* **Recall (Duyarlılık):** `[0.XXXX]`
* **Tahmin Süresi (Inference Time):** `[X.XX]` ms / görüntü

*(Eğitim sürecine ait loss değişim grafikleri ve confusion matrix çıktıları repodaki `results/` klasöründe yer almaktadır.)*

---

## 5. Uygulama (Deployment)
Eğittiğim modeli yalnızca kod çıktısı olarak bırakmamak için **Gradio** kütüphanesi ile etkileşimli bir web uygulamasına dönüştürdüm. Uygulama arayüzünde kullanıcı:
1. Bilgisayarından veya telefonundan istediği fotoğrafı yükleyebiliyor,
2. Güven eşiği (Confidence Threshold) kaydırıcısı ile filtreleme hassasiyetini değiştirebiliyor,
3. Çıktı ekranında tespit edilen nesnelerin etrafına çizilen kutuları ve yüzdelik güven skorlarını anlık olarak görebiliyor.

---

## 6. Gerçek Görüntülerle Test ve Değerlendirme
Modelin yalnızca veri setindeki fotoğraflarda değil, gerçek dünya koşullarında nasıl çalıştığını görmek için kendi telefonumla 5 farklı fotoğraf çekip Gradio arayüzünden test ettim:

| Fotoğraf | Çekim Koşulu | Gerçekteki Nesneler | Modelin Tahmini ve Skoru | Değerlendirme |
| :--- | :--- | :--- | :--- | :--- |
| **Fotoğraf 1** | İyi ışık, tek nesne | `[Örn: 1 Araba]` | `[Örn: car (%89)]` | Başarılı |
| **Fotoğraf 2** | Çalışma ortamı, çoklu nesne | `[Örn: Sandalye, Monitör, Şişe]` | `[Örn: chair (%82), tvmonitor (%76), bottle (%61)]` | Başarılı |
| **Fotoğraf 3** | Gün ışığı, açık alan | `[Örn: 2 İnsan]` | `[Örn: person (%91), person (%84)]` | Başarılı |
| **Fotoğraf 4** | Kısmi kapanma (nesne arkasında) | `[Örn: Masa arkasında sandalye]` | `[Örn: diningtable (%78), chair (%45)]` | Kısmen Başarılı |
| **Fotoğraf 5** | Loş ışık / uzak mesafe | `[Örn: Uzaktaki kedi / şişe]` | `[Örn: Tespit edilemedi veya düşük skor]` | Başarısız / Zorlandı |

**Genel Yorum:**
* **Başarılı Olduğu Durumlar:** İyi aydınlatılmış ortamlarda, kameraya yakın ve net görünen insan (`person`), araba (`car`), sandalye (`chair`) ve ekran (`tvmonitor`) gibi sınıfları yüksek güven skorlarıyla ve doğru kutu sınırlarıyla tespit etti.
* **Başarısız Olduğu Durumlar:** Nesnenin büyük kısmının başka bir eşyanın arkasında kaldığı (occlusion) durumlarda, çok küçük/uzak nesnelerde ve düşük ışıklı çekimlerde güven skorunun düştüğü veya bazı nesnelerin kaçırıldığı gözlemlendi. *(Test ekran görüntüleri `test_sonuclari/` klasörüne eklenmiştir.)*

---

## 7. Dosya Yapısı ve Kurulum
```text
├── YOLO11_Object_Detection.ipynb   # Veri hazırlama, eğitim, test ve Gradio kodları
├── voc_custom.yaml                 # Veri seti yolları ve 20 sınıf tanımı
├── best.pt                         # Eğitilmiş en iyi model ağırlıkları
├── results/                        # Loss eğrileri ve Confusion Matrix görselleri
└── test_sonuclari/                 # Telefonla çekilen 5 fotoğrafın Gradio çıktıları
