# YOLO11s ile Çok Sınıflı Nesne Tespiti (PASCAL VOC)

Bu proje, derin öğrenme tabanlı **YOLO11s** mimarisi kullanılarak 20 farklı nesne sınıfını gerçek zamanlı tespit etmek, model performansını bağımsız test verileriyle değerlendirmek ve eğitilen modeli **Gradio** web arayüzü üzerinden yayına almak (deployment) amacıyla geliştirilmiştir.

---

## 1. Veri Seti ve Ön İşleme

* **Veri Seti:** PASCAL VOC 2007 + 2012 Birleşik Veri Seti (Toplam **21.503** görüntü)
* **Sınıf Sayısı:** 20 Sınıf (`aeroplane`, `bicycle`, `bird`, `boat`, `bottle`, `bus`, `car`, `cat`, `chair`, `cow`, `diningtable`, `dog`, `horse`, `motorbike`, `person`, `pottedplant`, `sheep`, `sofa`, `train`, `tvmonitor`)
* **Veri Bölme Stratejisi (`seed=42`):** Veri sızıntısını önlemek amacıyla tüm görüntüler rastgele karıştırılarak üç bağımsız kümeye ayrılmıştır:
  * **Eğitim (Train - %70):** 15.052 görüntü
  * **Doğrulama (Validation - %15):** 3.225 görüntü
  * **Test (Test - %15):** 3.226 görüntü

---

## 2. Model Mimarisi ve Hiperparametreler

Model eğitimi Google Colab **NVIDIA Tesla T4 GPU** ortamında Ultralytics kütüphanesi ile gerçekleştirilmiştir.

| Parametre | Değer | Açıklama |
| :--- | :--- | :--- |
| **Mimari** | `YOLO11s` | Geliştirilmiş C3k2 blokları ve C2PSA dikkat mekanizmasına sahip güncel nesne tespit mimarisi |
| **Giriş Boyutu (`imgsz`)** | `640 x 640` | Standart ağ giriş çözünürlüğü |
| **Epoch & Erken Durdurma** | `20` (`patience=8`) | 8 epoch boyunca doğrulama başarısı artmazsa aşırı öğrenmeyi önlemek için erken durdurma |
| **Batch Size** | `32` | GPU bellek kapasitesine göre optimize edilmiş yığın boyutu |
| **Optimizer & Öğrenme Oranı** | `AdamW` (`lr0=0.001`) | Ağırlık çürümesi (weight decay) destekli adaptif optimizasyon |
| **Kayıp Fonksiyonları** | `CIoU` + `DFL` + `BCE` | Kutu konumlandırma (Box/Distribution Focal Loss) ve sınıflandırma (Binary Cross-Entropy) kayıpları |
| **Aktivasyon Fonksiyonu** | `SiLU` / `Sigmoid` | Gizli katmanlarda SiLU, çıkış katmanında Sigmoid aktivasyonu |
| **Rastgelelik Tohumu (`seed`)** | `123` | Tekrarlanabilir eğitim deneyleri için sabitlenen tohum değeri |

---

## 3. Test Kümesi Performans Sonuçları

Eğitim sürecinde modelin hiç görmediği **3.226 adet test görüntüsü** (`split="test"`) üzerinde elde edilen metrikler aşağıdadır:

* **Toplam Eğitim Süresi:** `[XX.XX]` dakika
* **mAP@0.5:** `[0.XXXX]`
* **mAP@0.5:0.95:** `[0.XXXX]`
* **Precision (Kesinlik):** `[0.XXXX]`
* **Recall (Duyarlılık):** `[0.XXXX]`
* **Çıkarım Hızı (Inference Speed):** `[X.XX]` ms / görüntü

---

## 4. Depo (Repository) İçeriği

* `YOLO11_Object_Detection.ipynb`: Veri hazırlama, model eğitimi, test değerlendirmesi ve Gradio dağıtım kodlarını içeren Google Colab defteri.
* `voc_custom.yaml`: Veri seti dizin yollarını ve 20 sınıf etiketini tanımlayan yapılandırma dosyası.
* `best.pt`: Doğrulama kümesinde en yüksek mAP skorunu elde eden eğitilmiş model ağırlıkları.
* `results/`: Eğitim sürecine ait kayıp (loss) grafikleri, karmaşıklık matrisi (confusion matrix) ve doğrulama tahmin görselleri.

---

## 5. Kurulum ve Web Arayüzünü Çalıştırma

Eğitilmiş modeli yerel ortamda veya Google Colab üzerinde Gradio arayüzü ile başlatmak için:

```bash
pip install ultralytics gradio
