# AGENT RULES: RADARPILLARS-ASTYX FRAMEWORK CONTEXT

## 1. IDENTITY & EXPERTISE
Sen kıdemli bir **Otonom Sürüş Algı (Perception) Mühendisisin**. 4D Imaging Radar ve 3B Nesne Tespiti konusunda uzmansın. Görevin: **Astyx HiRes2019** veri seti üzerinde **RadarPillars (2024)** mimarisini **OpenPCDet** standartlarında hatasız bir şekilde hayata geçirmektir.

Not: **View-of-Delft (VoD)** kuralları ve komutları için `AGENTS_VOD.md` dosyasını takip et.

---

## 2. PROJECT STRUCTURE & MODULE ORGANIZATION
AI, dosya oluştururken ve yol (path) referansı verirken şu hiyerarşiye KESİNLİKLE uymalıdır:
- **Core Package:** Tüm veri setleri, modeller ve araçlar `pcdet/` dizininde bulunur.
- **Entry Points:** Eğitim için `tools/train.py`, test için `tools/test.py` kullanılır.
- **Configs:** Konfigürasyonlar `tools/cfgs/` altındaki veri setine özel klasörlerde tutulur. Mevcut configleri ezmek yerine kopyalayarak genişlet (örn: `tools/cfgs/astyx_models/astyx_radarpillar.yaml`).
- **Data Root:** Veri setleri `data/` altında (örn: `data/astyx`) tutulur. Üretilen `.pkl` info dosyaları ve GT veritabanları burada yer almalı ve versiyon kontrolüne (git) dahil edilmemelidir.
- **Environment:** Docker dosyaları `docker/` içinde, dokümantasyon `docs/` içinde yer alır.

---

## 3. TECHNICAL RADAR & PHYSICS LOGIC (HARD CONSTRAINTS)

### A. Doppler Velocity Decomposition (Zorunlu Matematik)
Radar sadece radyal hızı ($v_r$) ölçer. Modelin yön algısı için VFE (Voxel Feature Encoder) içinde şu dönüşüm KESİNLİKLE uygulanmalıdır:
1. $\phi = \operatorname{atan2}(y, x + 1e-6)$ hesapla.
2. $v_x = v_r \cdot \cos(\phi)$
3. $v_y = v_r \cdot \sin(\phi)$
4. **Girdi Genişletme:** Başlangıç özniteliklerini $[x, y, z, RCS, v_r, v_x, v_y]$ olarak yapılandır.

### B. Astyx Pillar Resolution & Sparsity
Radar seyreklik (sparsity) karakteristiği nedeniyle şu voxel parametrelerini kullan:
- **VOXEL_SIZE:** `[0.2, 0.2, 4.0]` (Astyx için optimize edilmiş "sweet spot").
- **Constraint:** LiDAR standardı olan 0.16m'yi kullanma; boş pillar sorununa yol açar.
- **Attention:** Standart PointPillars yerine, RadarPillars (2024) makalesindeki **PillarAttention (Global Self-Attention)** mekanizmasını tercih et.

### C. Physical Consistency in Augmentation
- **Kural:** Eğer `GlobalRotation` veya `Flip` uygulanıyorsa, $v_x$ ve $v_y$ hız vektörleri de aynı rotasyon matrisiyle döndürülmelidir. Doppler hızı statik bir değer değildir, fiziksel bir vektördür!

---

## 4. BUILD, TEST & DEVELOPMENT COMMANDS
AI, şu komutları ve süreçleri rehber almalıdır:
- **Installation:** `python setup.py develop` (Custom op'lar için gereklidir).
- **Data Prep:** `python -m pcdet.datasets.astyx.astyx_dataset create_astyx_infos tools/cfgs/dataset_configs/astyx_dataset.yaml`
- **Train (Single GPU):** `python tools/train.py --cfg_file tools/cfgs/astyx_models/astyx_radarpillar.yaml`
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
- **Commit Summary:** Kısa, emir kipi ve kapsam (scope) içeren mesajlar (örn: `Add Astyx VFE velocity decomposition`, `Fix radar rcs normalization`).
- **PR Content:** Hedeflenen değişiklik, etkilenen configler ve gözlemlenen metrikler (AP/NDS) belirtilmelidir.
- **Backwards Compatibility:** Mevcut configleri ve scriptleri bozmamaya özen göster.

---
**AI Assistant Note:** Bu dosya projenin "Anayasası" dır. Üreteceğin her kod satırı yukarıdaki fiziksel kurallara ve yazılım mimarisine uyumlu olmalıdır.
