# YOLO11s ile Çok Sınıflı Nesne Tespiti (PASCAL VOC)

Bilgisayarlı görü dersi ödevi kapsamında, açık kaynaklı PASCAL VOC veri setini kullanarak 20 farklı nesne sınıfını konumlarıyla birlikte tespit eden (Object Detection) bir derin öğrenme modeli geliştirdim. Eğittiğim modeli eğitim sürecinde hiç görmediği bağımsız test verileriyle değerlendirdikten sonra, kullanıcıların kendi fotoğraflarını yükleyip anlık tespit alabilmesi için **Gradio** tabanlı bir web uygulamasına dönüştürdüm.

---

## Proje Raporu Özeti

| Alan | Açıklama / Öğrenci Yanıtı |
| :--- | :--- |
| **Project Title** | YOLO11s ile PASCAL VOC Veri Seti Üzerinde Çok Sınıflı Nesne Tespiti |
| **Problem / Task** | Nesne Tespiti (Object Detection — 20 Sınıf) |
| **Dataset** | PASCAL VOC 2007 + 2012 Birleşik Veri Seti (Toplam 21.503 görüntü: 15.052 Train, 3.225 Val, 3.226 Test) |
| **Dataset URL** | http://host.robots.ox.ac.uk/pascal/VOC/ |
| **Model Architecture** | YOLO11s (`yolo11s.pt`) |
| **Input Size** | 640 × 640 piksel |
| **Hyperparameters** | `epochs=20`, `batch=32`, `lr0=0.001`, `patience=8`, `save_period=5`, `cache=False`, `seed=123` |
| **Optimizer** | AdamW |
| **Activation / Output Function** | Gizli katmanlarda **SiLU**, sınıf olasılık çıkışlarında **Sigmoid** |
| **Loss Function** | **CIoU Loss** + **DFL** (kutu konumlandırma) ve **BCE Loss** (sınıflandırma) |
| **Evaluation Metrics** | IoU, Precision, Recall, mAP@0.5, mAP@0.5:0.95, Eğitim Süresi, Tahmin Süresi (ms) |
| **Training Environment** | Google Colab — NVIDIA Tesla T4 GPU (16 GB VRAM, 12.7 GB Sistem RAM), PyTorch & Ultralytics |
| **Best Performance** | Test mAP@0.5: **[0.XXXX]**, mAP@0.5:0.95: **[0.XXXX]** *(Eğitim Süresi: [XX.XX] dk)* |
| **Inference Time** | **[X.XX]** ms / görüntü |
| **Deployment Technology** | Gradio Web Arayüzü (`share=True`) |
| **Repository** | `[GitHub Repo Linkini Buraya Yapıştır]` |

---

## 1. Ödevin Amacı ve Seçilen Görev

Bu çalışmada, günlük yaşam sahnelerindeki birden fazla nesneyi hem sınıflandırabilen hem de görüntü üzerindeki koordinatlarını sınırlayıcı kutularla (bounding box) işaretleyebilen uçtan uca bir bilgisayarlı görü sistemi geliştirdim. Tek bir görüntüde farklı boyut ve konumlardaki nesneleri aynı anda yakalayabilmek amacıyla problem türü olarak **Nesne Tespiti (Object Detection)** görevini seçtim.

---

## 2. Veri Seti ve Ön İşleme Hattı

Model eğitiminde nesne tespiti literatüründe standart ölçüt kabul edilen **PASCAL VOC 2007 ve 2012** birleşik veri setini kullandım. Veri seti toplam **21.503 görüntü** ve **20 farklı sınıf** içermektedir:
`aeroplane`, `bicycle`, `bird`, `boat`, `bottle`, `bus`, `car`, `cat`, `chair`, `cow`, `diningtable`, `dog`, `horse`, `motorbike`, `person`, `pottedplant`, `sheep`, `sofa`, `train`, `tvmonitor`.

### Veri Sızıntısını Önleyen Bölme Stratejisi
Aynı görüntünün farklı kümelere düşmesini (data leakage) engellemek için tüm görüntü-etiket çiftlerini `seed=42` ile karıştırıp bağımsız üç kümeye ayırdım:
* **Eğitim Kümesi (Train — %70):** 15.052 görüntü
* **Doğrulama Kümesi (Validation — %15):** 3.225 görüntü
* **Test Kümesi (Test — %15):** 3.226 görüntü

```python
# Tekrarlanabilir ve sızıntısız 70 / 15 / 15 veri bölme yapısı
random.seed(42)
random.shuffle(tum_veriler)

n_toplam = len(tum_veriler)
n_train = int(n_toplam * 0.70)   # 15.052 görüntü
n_val = int(n_toplam * 0.15)     #  3.225 görüntü

train_verileri = tum_veriler[:n_train]
val_verileri = tum_veriler[n_train:n_train + n_val]
test_verileri = tum_veriler[n_train + n_val:]  # 3.226 görüntü
