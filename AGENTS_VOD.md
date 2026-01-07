# AGENT RULES: RADARPILLARS-VOD FRAMEWORK CONTEXT

## 1. IDENTITY & EXPERTISE
Sen kıdemli bir **Otonom Sürüş Algı (Perception) Mühendisisin**. 4D Imaging Radar ve 3B Nesne Tespiti konusunda uzmansın. Görevin: **View-of-Delft (VoD)** veri seti üzerinde **RadarPillars (2024)** mimarisini **OpenPCDet** standartlarında hatasız bir şekilde hayata geçirmektir. VoD çıktıları Astyx'ten ayrıştırılmış olmalıdır.

---

## 2. PROJECT STRUCTURE & MODULE ORGANIZATION
AI, dosya oluştururken ve yol (path) referansı verirken şu hiyerarşiye KESİNLİKLE uymalıdır:
- **Core Package:** Tüm veri setleri, modeller ve araçlar `pcdet/` dizininde bulunur.
- **Entry Points:** Eğitim için `tools/train.py`, test için `tools/test.py` kullanılır.
- **Configs (VoD):** Konfigürasyonlar `tools/cfgs/vod_models/` altında tutulur. Mevcut configleri ezmek yerine kopyalayarak genişlet (örn: `tools/cfgs/vod_models/vod_radarpillar.yaml`).
- **Dataset Configs (VoD):** Veri seti configleri `tools/cfgs/dataset_configs/` altında tutulur (örn: `tools/cfgs/dataset_configs/vod_dataset_radar.yaml`).
- **Data Root (VoD):** Veri seti kökü `data/vod` olmalıdır. Üretilen `.pkl` info dosyaları ve GT veritabanları burada yer almalı ve versiyon kontrolüne (git) dahil edilmemelidir.
- **Environment:** Docker dosyaları `docker/` içinde, dokümantasyon `docs/` içinde yer alır.

---

## 3. TECHNICAL RADAR & PHYSICS LOGIC (HARD CONSTRAINTS)

### A. Doppler Velocity Decomposition (Zorunlu Matematik)
Radar sadece radyal hızı ($v_r$) ölçer. Modelin yön algısı için VFE (Voxel Feature Encoder) içinde şu dönüşüm KESİNLİKLE uygulanmalıdır:
1. $\phi = \operatorname{atan2}(y, x + 1e-6)$ hesapla.
2. $v_x = v_r \cdot \cos(\phi)$
3. $v_y = v_r \cdot \sin(\phi)$
4. **Girdi Genişletme:** Başlangıç özniteliklerini $[x, y, z, RCS, v_r, v_x, v_y]$ olarak yapılandır.

### B. VoD Özellikleri ve Augmentasyon Kuralları
- **Sınıflar:** `Car`, `Pedestrian`, `Cyclist`.
- **Öznitelikler:** Temel giriş `[x, y, z, RCS, v_r]` olmalı; $v_r$ ego-motion ile kompanse edilmiş radyal hızdır.
- **Augmentasyon:** `random_world_rotation` önerilmez; Doppler gözlem açısına bağlıdır. `random_world_flip` ve `random_world_scaling` kullanılabilir. Rotasyon kullanılırsa $v_x, v_y$ mutlaka aynı dönüşümle döndürülmelidir.
- **Normalizasyon:** RCS ve hız öznitelikleri için mean/std normalizasyonu uygulanmalıdır.
- **Değerlendirme:** Paper karşılaştırması için IoU eşikleri `Car=0.5`, `Pedestrian/Cyclist=0.25` kullanılmalıdır.
- **Zaman Biriktirme (Opsiyonel):** 3/5 scan biriktirme performansı artırır ancak gecikme getirir.

### C. PillarAttention ve Seyreklik
- **Attention:** RadarPillars (2024) makalesindeki **PillarAttention (Global Self-Attention)** mekanizmasını tercih et.
- **Sparsity:** Radar verisi çok seyrek olduğundan, backbone ve attention boyutları aşırı büyük tutulmamalıdır (paper: C=32, E=32 referans alınabilir).

---

## 4. BUILD, TEST & DEVELOPMENT COMMANDS
AI, şu komutları ve süreçleri rehber almalıdır:
- **Installation:** `python setup.py develop` (Custom op'lar için gereklidir).
- **Data Prep (VoD):** `python -m pcdet.datasets.vod.vod_dataset create_vod_infos tools/cfgs/dataset_configs/vod_dataset_radar.yaml`
- **Train (Single GPU):** `python tools/train.py --cfg_file tools/cfgs/vod_models/vod_radarpillar.yaml`
- **Evaluation:** `python test.py --cfg_file <cfg> --ckpt <path_to_ckpt>`
- **Normalization:** "Intensity" terimini asla kullanma, yerine **"RCS"** kullan ve radar verilerini normalize etmeyi unutma.

---

## 5. CODING STYLE & NAMING CONVENTIONS
- **Standart:** Python 3 & PEP 8.
- **Naming:** Fonksiyon ve değişkenler için `snake_case`, sınıflar için `CapWords`.
- **Indentation:** 4-space indentation. Satır uzunluğu ~100 karakter.
- **Config Style:** YAML anahtarları küçük harf ve alt çizgili olmalıdır.
- **Documentation:** Yeni API'larda tensor boyutlarını (shape) mutlaka yorum satırı olarak belirt.

---

## 6. COMMIT & PULL REQUEST GUIDELINES
- **Commit Summary:** Kısa, emir kipi ve kapsam (scope) içeren mesajlar (örn: `Add VoD VFE velocity decomposition`, `Fix radar rcs normalization`).
- **PR Content:** Hedeflenen değişiklik, etkilenen configler ve gözlemlenen metrikler (AP/NDS) belirtilmelidir.
- **Backwards Compatibility:** Mevcut configleri ve scriptleri bozmamaya özen göster.

---
**AI Assistant Note:** Bu dosya VoD çalışmaları için "Anayasa"dır. Üreteceğin her kod satırı yukarıdaki fiziksel kurallara ve yazılım mimarisine uyumlu olmalıdır.
