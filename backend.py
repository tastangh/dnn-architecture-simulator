import numpy as np

def sigmoid(Z):
    """Sigmoid aktivasyon fonksiyonu."""
    # Sayısal kararlılık için çok büyük/küçük Z değerlerini kırp
    Z_clipped = np.clip(Z, -500, 500)
    A = 1 / (1 + np.exp(-Z_clipped))
    return A, Z  # Orijinal Z'yi cache için döndür

def relu(Z):
    """ReLU aktivasyon fonksiyonu."""
    A = np.maximum(0, Z)
    return A, Z  # Z'yi cache olarak döndür

def sigmoid_backward(dA, Z):
    """Sigmoid'in türevi."""
    s = 1 / (1 + np.exp(-np.clip(Z, -500, 500))) # Kırpılmış Z kullan
    dZ = dA * s * (1 - s)
    return dZ

def relu_backward(dA, Z):
    """ReLU'nun türevi."""
    dZ = np.array(dA, copy=True)  # dA'yı kopyala
    dZ[Z <= 0] = 0  # Z'nin 0'dan küçük veya eşit olduğu yerlerde gradyanı 0 yap
    return dZ

def calculate_loss(AL, Y, loss_type="mse"):
    """Kayıp fonksiyonunu hesaplar."""
    m = Y.shape[1]  # Örnek sayısı (genellikle 1 bu simülatörde)
    epsilon = 1e-8  # Log(0) veya bölme hatalarını önlemek için

    if loss_type == "cross_entropy":
        AL_clipped = np.clip(AL, epsilon, 1 - epsilon)
        cost = -np.sum(Y * np.log(AL_clipped) + (1 - Y) * np.log(1 - AL_clipped)) / m
        return np.squeeze(cost) # Skaler değer
    elif loss_type == "mae":
        return np.mean(np.abs(AL - Y))
    elif loss_type == "mse":
        return np.mean((AL - Y) ** 2)
    elif loss_type == "rmse":
        return np.sqrt(np.mean((AL - Y) ** 2) + epsilon) # Kök alırken 0 olmasını engelle
    else:
        raise ValueError(f"Desteklenmeyen kayıp fonksiyonu: {loss_type}")

def loss_gradient(AL, Y, loss_type="mse"):
    """Son katman aktivasyonuna (AL) göre kaybın türevini hesaplar."""
    m = Y.shape[1]
    epsilon = 1e-8

    if loss_type == "cross_entropy":
        AL_clipped = np.clip(AL, epsilon, 1 - epsilon)
        dAL = - (np.divide(Y, AL_clipped) - np.divide(1 - Y, 1 - AL_clipped)) / m
        return dAL
    elif loss_type == "mae":
        return np.sign(AL - Y) / m
    elif loss_type == "mse":
        return 2 * (AL - Y) / m
    elif loss_type == "rmse":
        mse = np.mean((AL - Y) ** 2)
        rmse = np.sqrt(mse + epsilon)
        return (AL - Y) / (m * rmse)
    else:
        raise ValueError(f"Desteklenmeyen kayıp fonksiyonu: {loss_type}")

class DeepDNN:
    """Derin Yapay Sinir Ağı Sınıfı."""
    def __init__(self, layer_sizes, learning_rate=0.01, hidden_activation="relu", loss_type="mse"):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.hidden_activation = hidden_activation
        self.loss_type = loss_type
        self.num_layers = len(layer_sizes) - 1
        self.parameters = {}
        self._initialize_parameters()

    def _initialize_parameters(self):
        """Ağırlık ve bias parametrelerini başlatır."""
        # np.random.seed(3) # Her seferinde aynı rastgeleliği kaldırdım, istenirse açılabilir
        for l in range(1, len(self.layer_sizes)):
            # He/Xavier başlatmaya benzer bir ölçeklendirme ekleyebiliriz (opsiyonel)
            scale_factor = np.sqrt(2. / self.layer_sizes[l-1]) if self.hidden_activation == "relu" else np.sqrt(1. / self.layer_sizes[l-1])
            self.parameters[f"W{l}"] = np.random.randn(self.layer_sizes[l], self.layer_sizes[l-1]) * scale_factor # * 0.01 yerine scale_factor
            self.parameters[f"b{l}"] = np.zeros((self.layer_sizes[l], 1))

    def _activation_forward(self, A_prev, W, b, activation):
        """Tek bir katman için ileri yayılım."""
        Z = np.dot(W, A_prev) + b
        if activation == "sigmoid":
            A, activation_cache = sigmoid(Z)
        elif activation == "relu":
            A, activation_cache = relu(Z)
        else:
            raise ValueError("Desteklenmeyen aktivasyon fonksiyonu")
        cache = (A_prev, W, b, activation_cache) # Z'yi saklar
        return A, cache

    def forward(self, X):
        """Tüm ağ için ileri yayılım."""
        caches = []
        A = X
        # Gizli katmanlar
        for l in range(1, self.num_layers):
            A_prev = A
            A, cache = self._activation_forward(A_prev, self.parameters[f"W{l}"], self.parameters[f"b{l}"], self.hidden_activation)
            caches.append(cache)
        # Çıkış katmanı (Sigmoid)
        # TODO: Çıkış aktivasyonunu da seçilebilir yapmak iyi olurdu
        AL, cache = self._activation_forward(A, self.parameters[f"W{self.num_layers}"], self.parameters[f"b{self.num_layers}"], "sigmoid")
        caches.append(cache)
        return AL, caches

    def _activation_backward(self, dA, cache, activation):
        """Tek bir katman için geri yayılım."""
        A_prev, W, b, Z = cache
        if activation == "relu":
            dZ = relu_backward(dA, Z)
        elif activation == "sigmoid":
            dZ = sigmoid_backward(dA, Z)
        else:
            raise ValueError("Desteklenmeyen aktivasyon fonksiyonu")
        m = A_prev.shape[1]
        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(W.T, dZ)
        return dA_prev, dW, db

    def backward(self, AL, Y, caches):
        """Tüm ağ için geri yayılım."""
        grads = {}
        L = self.num_layers
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)
        dAL = loss_gradient(AL, Y, self.loss_type)
        # Son katman (Sigmoid)
        current_cache = caches[L-1]
        grads[f"dA{L-1}"], grads[f"dW{L}"], grads[f"db{L}"] = self._activation_backward(dAL, current_cache, "sigmoid")
        # Diğer katmanlar
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
            # Gradyanların varlığını kontrol et (emin olmak için)
            if f"dW{l}" in grads and f"db{l}" in grads:
                self.parameters[f"W{l}"] -= self.learning_rate * grads[f"dW{l}"]
                self.parameters[f"b{l}"] -= self.learning_rate * grads[f"db{l}"]
            else:
                 print(f"Uyarı: Katman {l} için gradyanlar bulunamadı.")


    def train_step(self, X, Y):
        """Tek bir eğitim adımı."""
        AL, caches = self.forward(X)
        loss = calculate_loss(AL, Y, self.loss_type)
        grads = self.backward(AL, Y, caches)
        self.update_parameters(grads)
        return loss, AL

