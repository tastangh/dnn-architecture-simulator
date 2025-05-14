# DNN Simülatörü Örnek Çalıştırması: XOR Problemi

Bu dosya, DNN simülatörünün XOR problemi için tipik bir çalıştırma örneğini ve adım adım sonuçlarını göstermektedir.
Simülatördeki ilgili alanlara bu değerleri girerek benzer bir davranışı gözlemleyebilirsiniz.

## 1. Ağ Yapılandırması ve Hiperparametreler

Aşağıdaki ayarlar simülatör arayüzünden seçilmiştir:

*   **Input Yapılandırması:**
    *   Giriş Sayısı: `2`
*   **Output Yapılandırması:**
    *   Çıkış Sayısı: `1`
    *   Çıkış Aktivasyonu: `sigmoid`
*   **Gizli Katman Yapılandırması:**
    *   Gizli Katman Sayısı: `1`
    *   Gizli K. 1 Nöron: `4` (veya sizin kullandığınız değer)
    *   Gizli K. 1 Aktivasyon: `relu` (veya sizin kullandığınız değer)
*   **Eğitim Parametreleri:**
    *   Kayıp Fonksiyonu: `cross_entropy`
    *   Öğrenme Oranı: `0.1`
    *   Leaky ReLU Alpha: `0.01` (ReLU kullanıldığı için etkisiz, varsayılan)

## 2. Başlangıç Parametreleri (İsteğe Bağlı ama Çok Faydalı)

Eğer simülasyonu birebir tekrarlanabilir kılmak istiyorsanız, ilk adımdan önceki rastgele başlatılmış (veya manuel girilmiş) ağırlık ve biasları buraya ekleyebilirsiniz. Bu, `Ağırlık/Bias Parametre Girişi` özelliğiyle simülatöre yüklenebilir.

## 3. Adım Adım Simülasyon (XOR Örneği: x1=0, x2=0, Hedef y1=0)

Aşağıda, `Giriş (X): [0. 0.]` ve `Hedef (Y): [0.]` için ilk birkaç eğitim adımının simülatör çıktıları yer almaktadır.
Her adımdan sonra "Adım At (1 Kez Eğit)" butonuna basılmıştır.

**Başlangıç**
**Adım 1:**
[Adım 1] Giriş (X): [0. 0.]
Hedef (Y): [0.]
Tahmin (Y_pred): [0.5] // Rastgele ağırlıklarla tipik başlangıç
Kayıp (cross_entropy | Çıkış Akt: sigmoid): 0.693147


**Matematiksel Doğrulama Adımları:**

Adım A: İleri Yayılım (Forward Propagation)

Gizli Katman 1 (L1) Hesaplaması:

Pre-aktivasyon Z1 = W1 @ X + b1

X = [[0], [0]] olduğu için, W1 @ X her zaman bir sıfır vektörü ([[0],[0],[0],[0]]) olacaktır, W1'in değerleri ne olursa olsun.

Eğer b1 = [[0],[0],[0],[0]] (varsayımımız), o zaman:
Z1 = [[0],[0],[0],[0]] + [[0],[0],[0],[0]] = [[0],[0],[0],[0]]

Aktivasyon A1 = ReLU(Z1)

ReLU(0) = 0. Dolayısıyla,
A1 = [[0],[0],[0],[0]]

Çıkış Katmanı (L2) Hesaplaması:

Pre-aktivasyon Z2 = W2 @ A1 + b2

A1 = [[0],[0],[0],[0]] olduğu için, W2 @ A1 her zaman bir skaler sıfır ([[0]]) olacaktır, W2'nin değerleri ne olursa olsun.

Eğer b2 = [[0]] (varsayımımız), o zaman:
Z2 = [[0]] + [[0]] = [[0]]

Aktivasyon AL = sigmoid(Z2)

sigmoid(0) = 1 / (1 + exp(-0)) = 1 / (1 + 1) = 1 / 2 = 0.5

AL = [[0.5]]

Sonuç: Tahmin (Y_pred) = 0.5
Bu, simülatörünüzün verdiği Tahmin (Y_pred): [0.5] sonucuyla eşleşiyor!

Adım B: Kayıp Hesaplama (Loss Calculation)

Kayıp Fonksiyonu: cross_entropy (Binary Cross-Entropy)

AL = 0.5

Y_hedef = 0.0

Formül: Cost = - (Y_hedef * log(AL) + (1 - Y_hedef) * log(1 - AL))

Cost = - (0.0 * log(0.5) + (1 - 0.0) * log(1 - 0.5))

Cost = - (0 * log(0.5) + 1 * log(0.5))

Cost = - (0 + log(0.5))

Cost = - log(0.5)

Doğal logaritma ln(0.5) ≈ -0.69314718...

Cost ≈ - (-0.69314718) ≈ 0.69314718

Sonuç: Kayıp ≈ 0.693147
Bu, simülatörünüzün verdiği Kayıp (cross_entropy | Çıkış Akt: sigmoid): 0.693147 sonucuyla eşleşiyor!