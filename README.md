# Internship Assignments - Deep Learning Projects

Bu repository, derin öğrenme alanında gerçekleştirilen iki farklı assignment'ı içermektedir: görüntü sınıflandırma (computer vision) ve metin sınıflandırma (NLP) projeleri.

## 📁 Proje Yapısı

```
internship/
├── assignment1/          # Görüntü Sınıflandırma (CIFAR-10)
│   ├── assignment1_ft.py        # Fine-tuning implementasyonu
│   ├── assignment1_tl.py        # Transfer learning implementasyonu
│   ├── fine-tuning_results/     # Fine-tuning sonuçları
│   ├── transfer-learning_results/  # Transfer learning sonuçları
│   ├── data/                    # CIFAR-10 dataset
│   └── req.txt                  # Gerekli kütüphaneler
├── assignment2/          # Metin Sınıflandırma (Fake News Detection)
│   ├── assignment2.py           # NLP implementasyonu
│   ├── liar_dataset/           # LIAR dataset
│   ├── req.txt                 # Gerekli kütüphaneler
│   └── stats.txt               # Sonuçlar
└── README.md            # Bu dosya
```

## 🎯 Assignment 1: Görüntü Sınıflandırma (CIFAR-10)

### Proje Açıklaması
CIFAR-10 veri setinden **kedi (class 3)** ve **köpek (class 5)** sınıflarını kullanarak binary sınıflandırma yapan iki farklı yaklaşım:

1. **Fine-tuning**: Tüm ağın ağırlıkları güncellenir
2. **Transfer Learning**: Sadece son katmanın ağırlıkları güncellenir

### Metodoloji

#### Veri Hazırlama
- CIFAR-10'dan sadece kedi (3) ve köpek (5) sınıfları seçildi
- Label'lar yeniden etiketlendi: kedi=0, köpek=1
- Görüntüler 64x64 boyutuna yeniden boyutlandırıldı
- ImageNet ortalaması ve standart sapması ile normalize edildi

#### Model Mimarisi
- **Base Model**: ResNet18 (ImageNet pre-trained)
- **Fine-tuning**: Son katman 2 çıkışlı Linear layer ile değiştirildi
- **Transfer Learning**: Tüm katmanlar donduruldu, son katman çok katmanlı classifier ile değiştirildi.

#### Eğitim Parametreleri
- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss

### 📈 Görselleştirmeler
Her iki yaklaşım için şunlar oluşturuldu:
- Confusion Matrix
- Training/Validation Loss ve Accuracy grafikleri
- Doğru ve yanlış sınıflandırılan örnek görüntüler

### Çalıştırma
```bash
cd assignment1
pip install -r req.txt
python assignment1_ft.py      # Fine-tuning için
python assignment1_tl.py      # Transfer learning için
```

## 🎯 Assignment 2: Metin Sınıflandırma (Fake News Detection)

### Proje Açıklaması
LIAR veri seti kullanılarak sahte haber tespiti yapan NLP projesi. DistilBERT tabanlı model ile binary sınıflandırma gerçekleştirildi.

### Metodoloji

#### Veri Hazırlama
- **Dataset**: LIAR dataset (train.csv, test.csv, valid.csv)
- **Binary Labeling**: 
  - `false`, `pants-fire`, `barely-true` → 1 (sahte)
  - Diğer label'lar → 0 (gerçek)
- **Tokenization**: DistilBERT tokenizer

#### Model Mimarisi
- **Base Model**: DistilBERT (pre-trained, frozen)

#### Eğitim Stratejisi
- **Cross-Validation**: 5-fold StratifiedKFold
- **Optimizer**: AdamW
- **Loss Function**: NLLLoss (class weight balanced)

### Çalıştırma
```bash
cd assignment2
pip install -r req.txt
python assignment2.py
```

## 🔍 Karşılaştırma ve Analiz

### Assignment 1 Karşılaştırması
- **Fine-tuning** transfer learning'den önemli ölçüde daha iyi performans gösterdi
- Bu fark, tüm ağın domain-specific özellikler öğrenmesinden kaynaklanıyor

### Assignment 2 Challenges
- NLP görevi görüntü işlemeye göre daha zorlu
- Sahte haber tespiti subjektif ve karmaşık bir problem
- %62 accuracy makul bir sonuç (random guess %50)

## 🛠️ Gereksinimler

### Assignment 1
- torch
- torchvision
- numpy
- matplotlib
- scikit-learn
- seaborn

### Assignment 2
- torch
- numpy
- pandas
- transformers

## 📝 Notlar

- Her iki proje de GPU desteği içeriyor
- Sonuçlar reproducible için random seed'ler kullanıldı
- Detaylı metriks ve görselleştirmeler her proje için mevcut
- Model sonuçları ilgili klasörlerde saklanıyor

## 👤 Geliştirici
Bu projeler 2025 yılındaki ilk yaz stajı kapsamında derin öğrenme tekniklerini öğrenmek ve uygulamak amacıyla geliştirilmiştir. 