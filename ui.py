from PyQt5.QtWidgets import (
    QWidget, QLabel, QSpinBox, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QGridLayout, QTextEdit, QMessageBox, QGroupBox,
    QScrollArea, QDialog, QFormLayout, QDialogButtonBox, QComboBox,
    QDoubleSpinBox, QApplication
)

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch

from backend import DeepDNN
import numpy as np
import sys 

class DNNConfigurator(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DNN Architecture Simulator")
        self.resize(1200, 800)

        self.input_spin = QSpinBox()
        self.output_spin = QSpinBox()
        self.hidden_spin = QSpinBox()
        self.activation_combo = QComboBox()
        self.activation_combo.addItems(["relu", "sigmoid"])
        self.loss_combo = QComboBox()
        self.loss_combo.addItems(["mse", "mae", "rmse", "cross_entropy"])

        self.input_boxes = []
        self.hidden_layer_inputs = []
        self.output_boxes = []
        self.manual_param_widgets = {} 

        self.result_text = QTextEdit()
        self.network_config = None
        self.dnn = None 

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        # --- Input Yapılandırması ---
        input_group = QGroupBox("Input Yapılandırması")
        input_layout = QVBoxLayout()
        input_count_layout = QHBoxLayout()
        input_count_layout.addWidget(QLabel("Giriş Sayısı:"))
        self.input_spin.setMinimum(1)
        self.input_spin.setValue(3) # Başlangıç değeri
        self.input_spin.valueChanged.connect(self.update_input_boxes)
        input_count_layout.addWidget(self.input_spin)
        input_layout.addLayout(input_count_layout)
        self.input_grid = QGridLayout() # Giriş kutuları için grid
        input_layout.addLayout(self.input_grid)
        input_group.setLayout(input_layout)
        main_layout.addWidget(input_group)
        self.update_input_boxes()

        # --- Output Yapılandırması ---
        output_group = QGroupBox("Output Yapılandırması")
        output_layout = QVBoxLayout()
        output_count_layout = QHBoxLayout()
        output_count_layout.addWidget(QLabel("Çıkış Sayısı:"))
        self.output_spin.setMinimum(1)
        self.output_spin.setValue(1) # Başlangıç değeri
        self.output_spin.valueChanged.connect(self.update_output_boxes)
        output_count_layout.addWidget(self.output_spin)
        output_layout.addLayout(output_count_layout)
        self.output_grid = QGridLayout()
        output_layout.addLayout(self.output_grid)
        output_group.setLayout(output_layout)
        main_layout.addWidget(output_group)
        self.update_output_boxes() 

        # --- Hidden Layer Yapılandırması ---
        hidden_group = QGroupBox("Gizli Katman Yapılandırması")
        hidden_layout = QVBoxLayout()
        hidden_count_layout = QHBoxLayout()
        hidden_count_layout.addWidget(QLabel("Gizli Katman Sayısı:"))
        self.hidden_spin.setMinimum(0) 
        self.hidden_spin.setValue(1) 
        self.hidden_spin.valueChanged.connect(self.update_hidden_inputs)
        hidden_count_layout.addWidget(self.hidden_spin)
        hidden_layout.addLayout(hidden_count_layout)
        self.hidden_grid = QGridLayout() 
        hidden_layout.addLayout(self.hidden_grid)
        hidden_group.setLayout(hidden_layout)
        main_layout.addWidget(hidden_group)
        self.update_hidden_inputs()

        # --- Aktivasyon ve Kayıp Fonksiyonu ---
        act_loss_group = QGroupBox("Aktivasyon ve Kayıp Fonksiyonu")
        act_loss_layout = QHBoxLayout()
        act_loss_layout.addWidget(QLabel("Gizli Katman Aktivasyonu:"))
        act_loss_layout.addWidget(self.activation_combo)
        act_loss_layout.addWidget(QLabel("Kayıp Fonksiyonu:"))
        act_loss_layout.addWidget(self.loss_combo)
        act_loss_group.setLayout(act_loss_layout)
        main_layout.addWidget(act_loss_group)

        # --- Butonlar ---
        button_layout = QHBoxLayout()
        param_btn = QPushButton("Ağırlık/Bias Parametre Girişi") 
        param_btn.clicked.connect(self.enter_parameters) 
        button_layout.addWidget(param_btn)

        diagram_btn = QPushButton("Ağ Yapısını Görselleştir")
        diagram_btn.clicked.connect(self.visualize_architecture) 
        button_layout.addWidget(diagram_btn)

        run_btn = QPushButton("Adım At (Eğit)")
        run_btn.clicked.connect(self.run_step) 
        button_layout.addWidget(run_btn)
        main_layout.addLayout(button_layout)

        # --- Sonuçlar Alanı ---
        result_group = QGroupBox("Sonuçlar")
        result_layout = QVBoxLayout()
        self.result_text.setReadOnly(True)
        self.result_text.setFixedHeight(150) 
        result_layout.addWidget(self.result_text)
        result_group.setLayout(result_layout)
        main_layout.addWidget(result_group)

        # --- Kaydırılabilir Alan ---
        container = QWidget()
        container.setLayout(main_layout)
        scroll = QScrollArea()
        scroll.setWidget(container)
        scroll.setWidgetResizable(True)

        window_layout = QVBoxLayout(self)
        window_layout.addWidget(scroll)
        self.setLayout(window_layout)

    def clear_grid_layout(self, grid_layout):
        """Verilen QGridLayout'un içeriğini temizler."""
        while grid_layout.count():
            item = grid_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

    def update_input_boxes(self):
        """Giriş sayısı değiştikçe giriş kutularını günceller."""
        self.clear_grid_layout(self.input_grid)
        self.input_boxes = []
        cols = 4
        for i in range(self.input_spin.value()):
            label = QLabel(f"x{i+1}:")
            edit = QLineEdit("0.0")
            edit.setFixedWidth(60)
            row = i // cols
            col = (i % cols) * 2
            self.input_grid.addWidget(label, row, col)
            self.input_grid.addWidget(edit, row, col + 1)
            self.input_boxes.append(edit)
        self.dnn = None 

    def update_output_boxes(self):
        """Çıkış sayısı değiştikçe hedef çıkış kutularını günceller."""
        self.clear_grid_layout(self.output_grid)
        self.output_boxes = []
        cols = 4
        for i in range(self.output_spin.value()):
            label = QLabel(f"Hedef y{i+1}:")
            edit = QLineEdit("1.0")
            edit.setFixedWidth(60)
            row = i // cols
            col = (i % cols) * 2
            self.output_grid.addWidget(label, row, col)
            self.output_grid.addWidget(edit, row, col + 1)
            self.output_boxes.append(edit)
        self.dnn = None 

    def update_hidden_inputs(self):
        """Gizli katman sayısı değiştikçe nöron sayısı girişlerini günceller."""
        self.clear_grid_layout(self.hidden_grid)
        self.hidden_layer_inputs = []
        cols = 2
        for i in range(self.hidden_spin.value()):
            label = QLabel(f"Gizli Katman {i+1} Nöron Sayısı:")
            spin = QSpinBox()
            spin.setMinimum(1)
            spin.setValue(4)
            spin.setFixedWidth(80)
            row = i // cols
            col = (i % cols) * 2
            self.hidden_grid.addWidget(label, row, col)
            self.hidden_grid.addWidget(spin, row, col + 1)
            self.hidden_layer_inputs.append(spin)
        self.dnn = None 

    # --- YENİ PARAMETRE GİRİŞ FONKSİYONU ---
    def enter_parameters(self):
        """Her bir ağırlık ve bias için ayrı giriş alanı sunan diyalog açar."""
        try:
            # Mevcut ağ yapısını UI'dan al
            num_inputs = self.input_spin.value()
            num_outputs = self.output_spin.value()
            hidden_nodes = [spin.value() for spin in self.hidden_layer_inputs]
            layer_sizes = [num_inputs] + hidden_nodes + [num_outputs]

            if len(layer_sizes) < 2:
                 QMessageBox.warning(self, "Hata", "Parametre girmek için geçerli bir ağ yapısı tanımlanmalı (en az giriş ve çıkış katmanı).")
                 return

            # Ana diyalog penceresi
            dialog = QDialog(self)
            dialog.setWindowTitle("Parametre Girişi (Tek Tek)")
            dialog.setMinimumWidth(600) 
            dialog.setMinimumHeight(400)

            scroll_area = QScrollArea(dialog)
            scroll_area.setWidgetResizable(True)

            scroll_content = QWidget()
            params_layout = QVBoxLayout(scroll_content)

            self.manual_param_widgets = {}

            for l in range(len(layer_sizes) - 1):
                layer_index = l + 1 
                prev_layer_size = layer_sizes[l]
                current_layer_size = layer_sizes[l+1]

                layer_group = QGroupBox(f"Katman {layer_index} Parametreleri (L{l} -> L{layer_index})")
                layer_grid = QGridLayout() 

                # --- Ağırlıklar (W) ---
                w_key = f"W{layer_index}"
                self.manual_param_widgets[w_key] = [] 
                layer_grid.addWidget(QLabel(f"<b>Ağırlıklar (W{layer_index})</b> [Hedef Nörondan <- Kaynak Nörona]:"), 0, 0, 1, 4) 

                row_offset = 1 
                col_width = 2 

                for j in range(current_layer_size): # Hedef nöron (bu katman)
                    w_row_widgets = []
                    for k in range(prev_layer_size): # Kaynak nöron (önceki katman)
                        label = QLabel(f"W[{j+1}<-{k+1}]:")
                        spinbox = QDoubleSpinBox()
                        spinbox.setRange(-100.0, 100.0)
                        spinbox.setDecimals(4)        
                        spinbox.setSingleStep(0.01)
                        spinbox.setFixedWidth(100)

                        # Eğer DNN varsa ve parametreler mevcutsa, varsayılan değeri ata
                        default_val = 0.1 
                        if self.dnn and w_key in self.dnn.parameters:
                             if j < self.dnn.parameters[w_key].shape[0] and k < self.dnn.parameters[w_key].shape[1]:
                                 default_val = self.dnn.parameters[w_key][j, k]
                        spinbox.setValue(default_val)

                        # Grid'e ekle
                        grid_row = row_offset + j
                        grid_col = k * col_width
                        layer_grid.addWidget(label, grid_row, grid_col)
                        layer_grid.addWidget(spinbox, grid_row, grid_col + 1)
                        w_row_widgets.append(spinbox) 
                    self.manual_param_widgets[w_key].append(w_row_widgets) 

                # --- Biaslar (b) ---
                b_key = f"b{layer_index}"
                self.manual_param_widgets[b_key] = [] # Bu katmanın bias widget listesi
                bias_row_start = row_offset + current_layer_size + 1 # Ağırlıklardan sonra boşluk bırak
                layer_grid.addWidget(QLabel(f"<b>Biaslar (b{layer_index})</b> [Nöron]:"), bias_row_start -1 , 0, 1, 4)

                for j in range(current_layer_size): # Nöron (bu katman)
                    label = QLabel(f"b[{j+1}]:")
                    spinbox = QDoubleSpinBox()
                    spinbox.setRange(-100.0, 100.0)
                    spinbox.setDecimals(4)
                    spinbox.setSingleStep(0.01)
                    spinbox.setFixedWidth(100)

                    # Varsayılan değeri ata
                    default_val = 0.0 # Genel varsayılan
                    if self.dnn and b_key in self.dnn.parameters:
                        if j < self.dnn.parameters[b_key].shape[0]:
                           default_val = self.dnn.parameters[b_key][j, 0]
                    spinbox.setValue(default_val)

                    # Grid'e ekle (Ağırlıkların altına)
                    grid_row = bias_row_start + j
                    # Genellikle biasları tek sütunda göstermek yeterli
                    layer_grid.addWidget(label, grid_row, 0)
                    layer_grid.addWidget(spinbox, grid_row, 1)
                    self.manual_param_widgets[b_key].append(spinbox) 

                layer_group.setLayout(layer_grid)
                params_layout.addWidget(layer_group) # Grubu ana layout'a ekle

            # ScrollArea'nın içeriğini ayarla
            scroll_area.setWidget(scroll_content)

            buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            buttons.accepted.connect(dialog.accept)
            buttons.rejected.connect(dialog.reject)

            # Ana diyalog layout'u
            dialog_layout = QVBoxLayout(dialog)
            dialog_layout.addWidget(scroll_area) 
            dialog_layout.addWidget(buttons)    

            if dialog.exec_() == QDialog.Accepted:
                # Kullanıcı OK'a bastı, değerleri al ve DNN'e uygula
                new_params = {}
                try:
                    for l in range(len(layer_sizes) - 1):
                        layer_idx = l + 1
                        w_key = f"W{layer_idx}"
                        b_key = f"b{layer_idx}"
                        rows = layer_sizes[layer_idx]
                        cols = layer_sizes[l]

                        # Ağırlıkları topla
                        W = np.zeros((rows, cols))
                        for j in range(rows):
                            for k in range(cols):
                                W[j, k] = self.manual_param_widgets[w_key][j][k].value()
                        new_params[w_key] = W

                        # Biasları topla
                        b = np.zeros((rows, 1))
                        for j in range(rows):
                            b[j, 0] = self.manual_param_widgets[b_key][j].value()
                        new_params[b_key] = b

                    if self.dnn is None:
                         self.dnn = DeepDNN(layer_sizes,
                                            learning_rate=0.05, 
                                            hidden_activation=self.activation_combo.currentText(),
                                            loss_type=self.loss_combo.currentText())
                         self.result_text.append("Yeni DNN modeli oluşturuldu (parametre girişi sonrası).")

                    self.dnn.parameters = new_params
                    self.dnn.layer_sizes = layer_sizes 
                    QMessageBox.information(self, "Başarılı", "Girilen ağırlık ve bias değerleri ağa başarıyla yüklendi.")
                    self.result_text.append("Manuel parametreler ağa yüklendi.")
                    self.manual_param_widgets = {}

                except Exception as parse_error:
                     import traceback
                     traceback.print_exc()
                     QMessageBox.critical(self, "Parametre Yükleme Hatası", f"Parametreler ağa yüklenirken hata oluştu: {parse_error}")
            else:
                self.manual_param_widgets = {}

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.warning(self, "Parametre Girişi Hatası", f"Diyalog oluşturulurken hata: {str(e)}")


    def visualize_architecture(self):
        """Ağ mimarisini, ağırlık ve biasları görselleştirir."""
        try:
            params = None
            layer_sizes = []

            if self.dnn is None:
                QMessageBox.information(self, "Bilgi",
                                        "Ağırlık ve Biasları görmek için önce 'Adım At' butonuna basarak veya parametre girişi yaparak ağı başlatın.\nŞu an sadece yapı gösteriliyor.")
                num_inputs = self.input_spin.value()
                num_outputs = self.output_spin.value()
                hidden_nodes = [spin.value() for spin in self.hidden_layer_inputs]
                layer_sizes = [num_inputs] + hidden_nodes + [num_outputs]
                params = None 
            else:
                layer_sizes = self.dnn.layer_sizes
                params = self.dnn.parameters # Mevcut parametreleri al

            if not layer_sizes or len(layer_sizes) < 2:
                 QMessageBox.warning(self, "Hata", "Görselleştirilecek geçerli bir ağ yapısı yok.")
                 return

            max_neurons = max(layer_sizes) if layer_sizes else 1
            num_layers_viz = len(layer_sizes)

            # Figür boyutunu dinamik yap
            fig_width = max(12, num_layers_viz * 3.5) # Biraz daha geniş yapalım
            fig_height = max(8, max_neurons * 1.0) # Biraz daha yüksek yapalım
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))

            neuron_positions = []
            node_radius = 0.4 # Nöron yarıçapı
            h_spacing = 4.0 # Katmanlar arası yatay boşluk arttı
            v_spacing = 1.5 # Nöronlar arası dikey boşluk arttı

            # Nöronları çiz ve konumlarını sakla
            for i, layer_size in enumerate(layer_sizes):
                # Y ekseninde ortalamak için offset
                # Dikey boşluğu hesaba kat
                total_height = (layer_size - 1) * v_spacing
                y_start = (max_neurons * v_spacing - total_height) / 2.0 # Ortala

                layer_positions = []

                # Katman isimleri
                layer_name = ""
                if i == 0: layer_name = f"Giriş (L0)"
                elif i == num_layers_viz - 1: layer_name = f"Çıkış (L{i})"
                else: layer_name = f"Gizli {i} (L{i})"
                ax.text(i * h_spacing, y_start + total_height + v_spacing, f"{layer_name}\n({layer_size} nöron)",
                        ha='center', va='bottom', fontsize=9, weight='bold') # Başlığı yukarı taşı

                for j in range(layer_size):
                    x = i * h_spacing
                    y = y_start + j * v_spacing
                    circle = plt.Circle((x, y), node_radius, color='skyblue', zorder=4, ec='black') # Kenarlık
                    ax.add_patch(circle)
                    # Nöron numarasını yaz
                    ax.text(x, y, f"N{j+1}", ha='center', va='center', zorder=5, fontsize=7)
                    layer_positions.append({'pos': (x, y), 'id': j})

                    # Biasları ekle (giriş katmanı hariç ve parametreler varsa)
                    if params and i > 0:
                        bias_key = f"b{i}"
                        if bias_key in params and params[bias_key].shape[0] > j:
                            bias_val = params[bias_key][j, 0]
                            # Bias'ı nöronun biraz altına yaz
                            ax.text(x, y - node_radius * 1.5, f"b={bias_val:.2f}",
                                    ha='center', va='top', zorder=5, fontsize=6, color='red')

                neuron_positions.append(layer_positions)

            # Bağlantıları ve Ağırlıkları çiz
            if params: # Sadece parametreler varsa ağırlıkları çiz
                for l in range(num_layers_viz - 1):
                    weight_key = f"W{l+1}"
                    if weight_key in params:
                        W = params[weight_key] # W[hedef_nöron, kaynak_nöron]
                        for src_neuron in neuron_positions[l]:
                            for dst_neuron in neuron_positions[l + 1]:
                                src_id = src_neuron['id']
                                dst_id = dst_neuron['id']

                                if dst_id < W.shape[0] and src_id < W.shape[1]:
                                    src_pos = src_neuron['pos']
                                    dst_pos = dst_neuron['pos']
                                    weight_val = W[dst_id, src_id]

                                    arrow = FancyArrowPatch(src_pos, dst_pos, arrowstyle='->', mutation_scale=10,
                                                            color='gray', lw=0.5, zorder=1,
                                                            shrinkA=node_radius*1.2, shrinkB=node_radius*1.2)
                                    ax.add_patch(arrow)

                                    mid_x = src_pos[0] * 0.6 + dst_pos[0] * 0.4
                                    mid_y = src_pos[1] * 0.6 + dst_pos[1] * 0.4
                                    ax.text(mid_x, mid_y, f"w={weight_val:.2f}", fontsize=6, color='blue',
                                            ha='center', va='center', zorder=3,
                                            bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.7, ec='none'))
                                else:
                                    print(f"Uyarı: Ağırlık matrisi {weight_key} için indeks dışı: src={src_id}, dst={dst_id}. W shape: {W.shape}")
            else: # Parametre yoksa sadece okları çiz
                 for l in range(num_layers_viz - 1):
                     for src_neuron in neuron_positions[l]:
                         for dst_neuron in neuron_positions[l + 1]:
                             src_pos = src_neuron['pos']
                             dst_pos = dst_neuron['pos']
                             arrow = FancyArrowPatch(src_pos, dst_pos, arrowstyle='->', mutation_scale=10,
                                                     color='lightgray', lw=0.5, zorder=1,
                                                     shrinkA=node_radius*1.2, shrinkB=node_radius*1.2)
                             ax.add_patch(arrow)


            # Eksen ayarları
            ax.set_xlim(-node_radius*2, (num_layers_viz - 1) * h_spacing + node_radius*2)
            all_y = [n['pos'][1] for layer in neuron_positions for n in layer] if neuron_positions else [0]
            min_y_coord = min(all_y) - node_radius * 3
            max_y_coord = max(all_y) + node_radius * 2 + v_spacing # Başlık için yer
            # Eğer tek nöron varsa veya limitler çok yakınsa biraz boşluk bırak
            if max_y_coord - min_y_coord < 2:
                min_y_coord -= 1
                max_y_coord += 1

            ax.set_ylim(min_y_coord, max_y_coord)
            # ax.set_aspect('equal') # Bu, eksen limitleri ile çakışabilir, kaldırılabilir
            ax.axis('off')
            plt.title("DNN Mimarisi - Ağırlıklar ve Biaslar )", fontsize=12)
            plt.tight_layout(pad=2.0) # Kenarlarda boşluk bırak
            plt.show() # GUI'yi kilitler

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.warning(self, "Görselleştirme Hatası", f"Bir hata oluştu: {str(e)}")


    def run_step(self):
        """Tek bir eğitim adımı çalıştırır."""
        try:
            # Ağ yapılandırmasını al
            num_inputs = self.input_spin.value()
            num_outputs = self.output_spin.value()
            hidden_nodes = [spin.value() for spin in self.hidden_layer_inputs]
            current_config = [num_inputs] + hidden_nodes + [num_outputs]

            # Giriş (X) ve Hedef (Y) değerlerini al
            x_vals = []
            for box in self.input_boxes:
                try: x_vals.append(float(box.text()))
                except ValueError:
                    QMessageBox.warning(self, "Giriş Hatası", f"Geçersiz giriş değeri: '{box.text()}'."); return
            x = np.array(x_vals).reshape(-1, 1)

            y_vals = []
            for box in self.output_boxes:
                 try: y_vals.append(float(box.text()))
                 except ValueError:
                    QMessageBox.warning(self, "Hedef Çıkış Hatası", f"Geçersiz hedef çıkış değeri: '{box.text()}'."); return
            y = np.array(y_vals).reshape(-1, 1)

            # Girdi/çıktı sayısı kontrolü
            if len(x_vals) != num_inputs:
                QMessageBox.warning(self, "Yapılandırma Hatası", f"Giriş sayısı ({num_inputs}) ile girilen değer sayısı ({len(x_vals)}) uyuşmuyor."); return
            if len(y_vals) != num_outputs:
                 QMessageBox.warning(self, "Yapılandırma Hatası", f"Çıkış sayısı ({num_outputs}) ile girilen hedef değer sayısı ({len(y_vals)}) uyuşmuyor."); return

            activation = self.activation_combo.currentText()
            loss_type = self.loss_combo.currentText()


            if self.dnn is None or self.dnn.layer_sizes != current_config:
                self.dnn = DeepDNN(current_config, learning_rate=0.05,
                                hidden_activation=activation, loss_type=loss_type)
                self.result_text.append("Yeni DNN modeli oluşturuldu/yapılandırıldı ('Adım At' ile).")
                self.network_config = current_config 

            # Mevcut DNN'in aktivasyon/loss tipi UI ile uyuşmuyorsa güncelle
            elif self.dnn.hidden_activation != activation or self.dnn.loss_type != loss_type:
                self.dnn.hidden_activation = activation
                self.dnn.loss_type = loss_type
                # self.dnn.learning_rate = ... # Öğrenme oranını da UI'dan alıp güncelleyebiliriz
                self.result_text.append(f"Mevcut DNN için Aktivasyon '{activation}' ve Kayıp '{loss_type}' olarak güncellendi.")

            # Eğitim adımını çalıştır
            loss, y_pred = self.dnn.train_step(x, y)

            # Sonuçları göster
            self.result_text.append("-" * 20)
            self.result_text.append(f"Giriş (X): {x.ravel()}")
            self.result_text.append(f"Hedef (Y): {y.ravel()}")
            self.result_text.append(f"Tahmin (Y_pred): {np.round(y_pred.ravel(), 4)}")
            self.result_text.append(f"Kayıp ({loss_type}): {loss:.6f}")
            self.result_text.append("")
            self.result_text.verticalScrollBar().setValue(self.result_text.verticalScrollBar().maximum())

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Çalıştırma Hatası", f"Adım atılırken bir hata oluştu: {str(e)}")


# Ana uygulama kısmı (main.py'den alınabilir veya burada kalabilir)
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DNNConfigurator()
    window.show()
    sys.exit(app.exec_())

