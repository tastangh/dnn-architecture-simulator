# --- START OF FILE backend.py ---

import numpy as np

def sigmoid(Z):
    """Sigmoid aktivasyon fonksiyonu."""
    A = 1 / (1 + np.exp(-Z))
    return A, Z  # Z'yi cache olarak döndür

def relu(Z):
    """ReLU aktivasyon fonksiyonu."""
    A = np.maximum(0, Z)
    return A, Z  # Z'yi cache olarak döndür

def sigmoid_backward(dA, Z):
    """Sigmoid'in türevi."""
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    return dZ

def relu_backward(dA, Z):
    """ReLU'nun türevi."""
    dZ = np.array(dA, copy=True)  # dA'yı kopyala
    dZ[Z <= 0] = 0  # Z'nin 0'dan küçük veya eşit olduğu yerlerde gradyanı 0 yap
    return dZ

def calculate_loss(AL, Y, loss_type="mse"):
    """Kayıp fonksiyonunu hesaplar."""
    m = Y.shape[1]  # Örnek sayısı
    epsilon = 1e-8  # Log(0) veya bölme hatalarını önlemek için küçük bir değer

    if loss_type == "cross_entropy":
        # Olasılıkları 0 veya 1 olmaktan koru
        AL_clipped = np.clip(AL, epsilon, 1 - epsilon)
        # Cross-entropy kaybı
        cost = -np.sum(Y * np.log(AL_clipped) + (1 - Y) * np.log(1 - AL_clipped)) / m
        return np.squeeze(cost) # Skaler değer döndür
    elif loss_type == "mae":
        # Ortalama Mutlak Hata (Mean Absolute Error)
        return np.mean(np.abs(AL - Y))
    elif loss_type == "mse":
        # Ortalama Kare Hata (Mean Squared Error)
        return np.mean((AL - Y) ** 2)
    elif loss_type == "rmse":
        # Kök Ortalama Kare Hata (Root Mean Squared Error)
        return np.sqrt(np.mean((AL - Y) ** 2))
    else:
        raise ValueError(f"Desteklenmeyen kayıp fonksiyonu: {loss_type}")

def loss_gradient(AL, Y, loss_type="mse"):
    """Son katman aktivasyonuna (AL) göre kaybın türevini hesaplar."""
    m = Y.shape[1]
    epsilon = 1e-8

    if loss_type == "cross_entropy":
        # Cross-entropy kaybının AL'ye göre türevi
        AL_clipped = np.clip(AL, epsilon, 1 - epsilon)
        dAL = - (np.divide(Y, AL_clipped) - np.divide(1 - Y, 1 - AL_clipped)) / m
        return dAL
    elif loss_type == "mae":
        # MAE'nin AL'ye göre türevi
        return np.sign(AL - Y) / m
    elif loss_type == "mse":
        # MSE'nin AL'ye göre türevi
        return 2 * (AL - Y) / m
    elif loss_type == "rmse":
        # RMSE'nin AL'ye göre türevi
        mse = np.mean((AL - Y) ** 2)
        rmse = np.sqrt(mse + epsilon) # 0'a bölme hatasını önle
        return (AL - Y) / (m * rmse)
    else:
        raise ValueError(f"Desteklenmeyen kayıp fonksiyonu: {loss_type}")

class DeepDNN:
    """Derin Yapay Sinir Ağı Sınıfı."""
    def __init__(self, layer_sizes, learning_rate=0.01, hidden_activation="relu", loss_type="mse"):
        """
        Başlatıcı.
        Args:
            layer_sizes (list): Her katmandaki nöron sayısını içeren liste (giriş dahil).
            learning_rate (float): Öğrenme oranı.
            hidden_activation (str): Gizli katmanlar için aktivasyon fonksiyonu ("relu" veya "sigmoid").
            loss_type (str): Kullanılacak kayıp fonksiyonu ("mse", "mae", "rmse", "cross_entropy").
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.hidden_activation = hidden_activation
        self.loss_type = loss_type
        self.num_layers = len(layer_sizes) - 1 # Katman sayısı (giriş hariç)
        self.parameters = {}
        self._initialize_parameters()

    def _initialize_parameters(self):
        """Ağırlık ve bias parametrelerini başlatır."""
        np.random.seed(3) # Tekrarlanabilirlik için
        for l in range(1, len(self.layer_sizes)):
            # Ağırlık matrisi (küçük rastgele değerler)
            self.parameters[f"W{l}"] = np.random.randn(self.layer_sizes[l], self.layer_sizes[l-1]) * 0.01
            # Bias vektörü (sıfırlar)
            self.parameters[f"b{l}"] = np.zeros((self.layer_sizes[l], 1))

    def _activation_forward(self, A_prev, W, b, activation):
        """Tek bir katman için ileri yayılım (lineer + aktivasyon)."""
        # Lineer kısım
        Z = np.dot(W, A_prev) + b

        # Aktivasyon kısmı
        if activation == "sigmoid":
            A, activation_cache = sigmoid(Z)
        elif activation == "relu":
            A, activation_cache = relu(Z)
        else:
            raise ValueError("Desteklenmeyen aktivasyon fonksiyonu")

        # Geri yayılım için gerekli değerleri sakla
        cache = (A_prev, W, b, activation_cache) # activation_cache burada Z'dir
        return A, cache

    def forward(self, X):
        """Tüm ağ için ileri yayılım."""
        caches = []
        A = X # Başlangıç aktivasyonu giriş verisidir

        # Gizli katmanlar (ReLU veya Sigmoid)
        for l in range(1, self.num_layers):
            A_prev = A
            A, cache = self._activation_forward(A_prev, self.parameters[f"W{l}"], self.parameters[f"b{l}"], self.hidden_activation)
            caches.append(cache)

        # Çıkış katmanı (Sigmoid - varsayılan)
        AL, cache = self._activation_forward(A, self.parameters[f"W{self.num_layers}"], self.parameters[f"b{self.num_layers}"], "sigmoid")
        caches.append(cache)

        return AL, caches

    def _activation_backward(self, dA, cache, activation):
        """Tek bir katman için geri yayılım (aktivasyon + lineer)."""
        A_prev, W, b, Z = cache # Cache'den değerleri al

        # Aktivasyonun türevi
        if activation == "relu":
            dZ = relu_backward(dA, Z)
        elif activation == "sigmoid":
            dZ = sigmoid_backward(dA, Z)
        else:
            raise ValueError("Desteklenmeyen aktivasyon fonksiyonu")

        # Lineer kısmın türevleri
        m = A_prev.shape[1] # Örnek sayısı
        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(W.T, dZ)

        return dA_prev, dW, db

    def backward(self, AL, Y, caches):
        """Tüm ağ için geri yayılım."""
        grads = {}
        L = self.num_layers # Katman sayısı
        m = AL.shape[1]
        Y = Y.reshape(AL.shape) # Y'nin AL ile aynı boyutta olduğundan emin ol

        # Son katmanın gradyanını başlat (kayıp fonksiyonunun türevi)
        dAL = loss_gradient(AL, Y, self.loss_type)

        # Son katman (Sigmoid) için geri yayılım
        current_cache = caches[L-1]
        grads[f"dA{L-1}"], grads[f"dW{L}"], grads[f"db{L}"] = self._activation_backward(dAL, current_cache, "sigmoid")

        # Diğer katmanlar için geri yayılım (ReLU veya Sigmoid)
        for l in reversed(range(L - 1)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self._activation_backward(grads[f"dA{l+1}"], current_cache, self.hidden_activation)
            grads[f"dA{l}"] = dA_prev_temp
            grads[f"dW{l+1}"] = dW_temp
            grads[f"db{l+1}"] = db_temp

        return grads

    def update_parameters(self, grads):
        """Gradyanları kullanarak parametreleri günceller."""
        L = self.num_layers
        for l in range(1, L + 1):
            self.parameters[f"W{l}"] = self.parameters[f"W{l}"] - self.learning_rate * grads[f"dW{l}"]
            self.parameters[f"b{l}"] = self.parameters[f"b{l}"] - self.learning_rate * grads[f"db{l}"]

    def train_step(self, X, Y):
        """Tek bir eğitim adımı (ileri yayılım, kayıp, geri yayılım, güncelleme)."""
        # İleri yayılım
        AL, caches = self.forward(X)

        # Kayıp hesaplama
        loss = calculate_loss(AL, Y, self.loss_type)

        # Geri yayılım
        grads = self.backward(AL, Y, caches)

        # Parametre güncelleme
        self.update_parameters(grads)

        return loss, AL

# --- END OF FILE backend.py ---