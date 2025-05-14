import numpy as np

def sigmoid(Z):
    """Sigmoid aktivasyon fonksiyonu."""
    Z_clipped = np.clip(Z, -500, 500)
    A = 1 / (1 + np.exp(-Z_clipped))
    return A, Z

def relu(Z):
    """ReLU aktivasyon fonksiyonu."""
    A = np.maximum(0, Z)
    return A, Z

def tanh(Z):
    """Tanh aktivasyon fonksiyonu."""
    A = np.tanh(Z)
    return A, Z

def leaky_relu(Z, alpha=0.01):
    """Leaky ReLU aktivasyon fonksiyonu."""
    A = np.where(Z > 0, Z, Z * alpha)
    activation_cache = (Z, alpha) 
    return A, activation_cache

def softmax(Z):
    """Softmax aktivasyon fonksiyonu."""
    Z_shifted = Z - np.max(Z, axis=0, keepdims=True)
    exp_Z = np.exp(Z_shifted)
    A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
    return A, A 

def linear(Z):
    """Lineer aktivasyon (aktivasyon yok)."""
    A = Z
    return A, Z

# --- Mevcut Geri Yayılım Fonksiyonları ---
def sigmoid_backward(dA, Z):
    """Sigmoid'in türevi."""
    s = 1 / (1 + np.exp(-np.clip(Z, -500, 500)))
    dZ = dA * s * (1 - s)
    return dZ

def relu_backward(dA, Z):
    """ReLU'nun türevi."""
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

# --- YENİ Geri Yayılım Fonksiyonları ---
def tanh_backward(dA, Z):
    """Tanh'ın türevi."""
    s = np.tanh(Z)
    dZ = dA * (1 - np.power(s, 2))
    return dZ

def leaky_relu_backward(dA, activation_cache):
    """Leaky ReLU'nun türevi."""
    Z, alpha = activation_cache 
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] *= alpha
    return dZ

def softmax_backward(dA, A_cache):
    """Softmax'ın türevi (dL/dZ). dA (dL/dA) ve A (softmax çıktısı) alır."""
    # Bu fonksiyon dL/dZ'yi hesaplar. Genellikle softmax + cross-entropy için
    # dL/dZ = A - Y kısayolu kullanılır. Bu fonksiyon daha genel bir durum içindir.
    S = A_cache # S, softmax çıktısıdır (AL)
    # dZ_i = S_i * (dA_i - sum_j(dA_j * S_j))
    dZ = S * (dA - np.sum(dA * S, axis=0, keepdims=True))
    return dZ

def linear_backward(dA, Z): # Z burada teknik olarak gerekmeyebilir ama imza tutarlılığı içi
    """Lineer aktivasyonun türevi."""
    dZ = dA
    return dZ


def calculate_loss(AL, Y, loss_type="mse"):
    m = Y.shape[1]
    epsilon = 1e-8

    if loss_type == "cross_entropy":
        # Bu binary cross-entropy veya Y one-hot ise categorical cross-entropy'ye yakınsar.
        # Softmax ile kullanıldığında, Y one-hot ise -np.sum(Y * np.log(AL_clipped)) / m daha yaygındır.
        # Şimdilik mevcut haliyle bırakıyoruz.
        AL_clipped = np.clip(AL, epsilon, 1 - epsilon)
        cost = -np.sum(Y * np.log(AL_clipped) + (1 - Y) * np.log(1 - AL_clipped)) / m
        return np.squeeze(cost)
    elif loss_type == "mae":
        return np.mean(np.abs(AL - Y))
    elif loss_type == "mse":
        return np.mean((AL - Y) ** 2)
    elif loss_type == "rmse":
        return np.sqrt(np.mean((AL - Y) ** 2) + epsilon)
    else:
        raise ValueError(f"Desteklenmeyen kayıp fonksiyonu: {loss_type}")

def loss_gradient(AL, Y, loss_type="mse"):
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
    def __init__(self, layer_sizes, learning_rate=0.01,
                 activations=None, # Tek bir liste: [hidden1_act, hidden2_act, ..., output_act]
                 loss_type="mse", leaky_relu_alpha=0.01):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.loss_type = loss_type
        self.leaky_relu_alpha = leaky_relu_alpha
        self.num_layers = len(layer_sizes) - 1 # Katman sayısı (W ve b parametre setlerinin sayısı)

        if activations is None:
            # Varsayılan: tüm gizliler relu, çıkış sigmoid
            self.activations = ["relu"] * (self.num_layers - 1) + ["sigmoid"]
        elif isinstance(activations, str): # Geriye dönük uyumluluk için tek string gelirse
            self.activations = [activations] * (self.num_layers - 1) + ["sigmoid"] # Veya çıkışı da aynı yap
        else:
            self.activations = activations

        if len(self.activations) != self.num_layers:
            raise ValueError(f"Aktivasyon listesi uzunluğu ({len(self.activations)}) "
                             f"katman sayısı ({self.num_layers}) ile eşleşmelidir.")

        self.parameters = {}
        self._initialize_parameters()

    def _initialize_parameters(self):
        for l in range(1, self.num_layers + 1): # l: 1'den num_layers'a kadar
            current_activation = self.activations[l-1] # l-1 çünkü self.activations 0-indeksli
            
            if current_activation == "relu" or current_activation == "leaky_relu":
                scale_factor = np.sqrt(2. / self.layer_sizes[l-1]) # He initialization
            elif current_activation == "tanh":
                 scale_factor = np.sqrt(1. / self.layer_sizes[l-1]) # Xavier/Glorot
            else: # sigmoid, softmax, linear için genel Xavier/Glorot
                scale_factor = np.sqrt(1. / self.layer_sizes[l-1])

            self.parameters[f"W{l}"] = np.random.randn(self.layer_sizes[l], self.layer_sizes[l-1]) * scale_factor
            self.parameters[f"b{l}"] = np.zeros((self.layer_sizes[l], 1))

    def _activation_forward(self, A_prev, W, b, activation_type):
        Z = np.dot(W, A_prev) + b
        if activation_type == "sigmoid":
            A, activation_cache = sigmoid(Z)
        elif activation_type == "relu":
            A, activation_cache = relu(Z)
        elif activation_type == "tanh":
            A, activation_cache = tanh(Z)
        elif activation_type == "leaky_relu":
            A, activation_cache = leaky_relu(Z, alpha=self.leaky_relu_alpha)
        elif activation_type == "softmax":
            A, activation_cache = softmax(Z)
        elif activation_type == "linear":
            A, activation_cache = linear(Z)
        else:
            raise ValueError(f"Desteklenmeyen aktivasyon fonksiyonu: {activation_type}")
        cache = (A_prev, W, b, activation_cache)
        return A, cache

    def forward(self, X):
        caches = []
        A = X
        for l in range(1, self.num_layers + 1): # l: 1'den num_layers'a kadar
            A_prev = A
            activation_func = self.activations[l-1] # Bu katmanın aktivasyonu
            A, cache = self._activation_forward(A_prev, self.parameters[f"W{l}"], self.parameters[f"b{l}"], activation_func)
            caches.append(cache)
        AL = A # Son katmanın çıktısı
        return AL, caches

    def _activation_backward(self, dA, cache, activation_type):
        A_prev, W, b, activation_cache_data = cache
        if activation_type == "relu":
            dZ = relu_backward(dA, activation_cache_data)
        elif activation_type == "sigmoid":
            dZ = sigmoid_backward(dA, activation_cache_data)
        elif activation_type == "tanh":
            dZ = tanh_backward(dA, activation_cache_data)
        elif activation_type == "leaky_relu":
            dZ = leaky_relu_backward(dA, activation_cache_data)
        elif activation_type == "softmax":
            dZ = softmax_backward(dA, activation_cache_data) # A (softmax çıktısı) alır
        elif activation_type == "linear":
            dZ = linear_backward(dA, activation_cache_data)
        else:
            raise ValueError(f"Desteklenmeyen aktivasyon fonksiyonu: {activation_type}")
        m = A_prev.shape[1]
        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(W.T, dZ)
        return dA_prev, dW, db

    def backward(self, AL, Y, caches):
        grads = {}
        L = self.num_layers # L, W ve b parametre setlerinin sayısıdır
        Y = Y.reshape(AL.shape)

        # Son katman (L-th layer, index L-1 in caches and activations)
        current_cache = caches[L-1]
        output_activation_func = self.activations[L-1]

        if output_activation_func == "softmax" and self.loss_type == "cross_entropy":
            dZL = AL - Y # dL/dZ_L
            A_prev_L_minus_1, WL, bL, _ = current_cache
            m = A_prev_L_minus_1.shape[1]
            grads[f"dW{L}"] = np.dot(dZL, A_prev_L_minus_1.T) / m
            grads[f"db{L}"] = np.sum(dZL, axis=1, keepdims=True) / m
            grads[f"dA{L-1}"] = np.dot(WL.T, dZL)
        else:
            dAL = loss_gradient(AL, Y, self.loss_type) # dL/dAL
            grads[f"dA{L-1}"], grads[f"dW{L}"], grads[f"db{L}"] = \
                self._activation_backward(dAL, current_cache, output_activation_func)

        # Diğer katmanlar (L-1 down to 1)
        # l burada `caches` ve `activations` için 0-indeksli katman numarasını (L-2'den 0'a) temsil eder
        # Parametreler (dW, db) ise l+1 katmanına ait olur. dA ise l katmanına aittir.
        for l in reversed(range(L - 1)): # l: L-2, L-3, ..., 0
            current_cache = caches[l]
            activation_func = self.activations[l] # Bu gizli katmanın aktivasyonu
            # grads[f"dA{l+1}"] bir sonraki (daha sağdaki) katmandan gelen dA_prev'dir
            dA_prev_temp, dW_temp, db_temp = \
                self._activation_backward(grads[f"dA{l+1}"], current_cache, activation_func)
            grads[f"dA{l}"] = dA_prev_temp       # Bu, l. katmanın (0-indeksli) aktivasyonuna göre gradyan
            grads[f"dW{l+1}"] = dW_temp     # Bu, (l+1). katmanın (1-indeksli) ağırlıklarına göre gradyan
            grads[f"db{l+1}"] = db_temp     # Bu, (l+1). katmanın (1-indeksli) biaslarına göre gradyan
            
        return grads

    def update_parameters(self, grads):
        L = self.num_layers
        for l in range(1, L + 1):
            if f"dW{l}" in grads and f"db{l}" in grads:
                self.parameters[f"W{l}"] -= self.learning_rate * grads[f"dW{l}"]
                self.parameters[f"b{l}"] -= self.learning_rate * grads[f"db{l}"]
            else:
                 print(f"Uyarı: Katman {l} için gradyanlar bulunamadı.")

    def train_step(self, X, Y):
        AL, caches = self.forward(X)
        loss = calculate_loss(AL, Y, self.loss_type)
        grads = self.backward(AL, Y, caches)
        self.update_parameters(grads)
        return loss, AL