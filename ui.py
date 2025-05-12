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
import traceback # Hata ayıklama için

class DNNConfigurator(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DNN Architecture Simulator")
        self.resize(1200, 800)

        # Arayüz Elemanları
        self.input_spin = QSpinBox()
        self.output_spin = QSpinBox()
        self.hidden_spin = QSpinBox()
        self.activation_combo = QComboBox()
        self.loss_combo = QComboBox()
        self.lr_spinbox = QDoubleSpinBox() # Öğrenme Oranı için
        self.steps_spinbox = QSpinBox()    # Adım Sayısı için

        # Diğer Üyeler
        self.input_boxes = []
        self.hidden_layer_inputs = []
        self.output_boxes = []
        self.manual_param_widgets = {}
        self.result_text = QTextEdit()
        self.network_config = None
        self.dnn = None
        self.current_x = None # Hazırlanan giriş verisi
        self.current_y = None # Hazırlanan hedef veri

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        # --- Yapılandırma Grupları (Input, Output, Hidden) ---
        # (Bu kısımlar önceki kodla aynı, değişiklik yok)
        # Input Group
        input_group = QGroupBox("Input Yapılandırması")
        input_layout = QVBoxLayout()
        input_count_layout = QHBoxLayout()
        input_count_layout.addWidget(QLabel("Giriş Sayısı:"))
        self.input_spin.setMinimum(1); self.input_spin.setValue(3)
        self.input_spin.valueChanged.connect(self.update_input_boxes)
        input_count_layout.addWidget(self.input_spin)
        input_layout.addLayout(input_count_layout)
        self.input_grid = QGridLayout(); input_layout.addLayout(self.input_grid)
        input_group.setLayout(input_layout); main_layout.addWidget(input_group)
        self.update_input_boxes()

        # Output Group
        output_group = QGroupBox("Output Yapılandırması")
        output_layout = QVBoxLayout()
        output_count_layout = QHBoxLayout()
        output_count_layout.addWidget(QLabel("Çıkış Sayısı:"))
        self.output_spin.setMinimum(1); self.output_spin.setValue(1)
        self.output_spin.valueChanged.connect(self.update_output_boxes)
        output_count_layout.addWidget(self.output_spin)
        output_layout.addLayout(output_count_layout)
        self.output_grid = QGridLayout(); output_layout.addLayout(self.output_grid)
        output_group.setLayout(output_layout); main_layout.addWidget(output_group)
        self.update_output_boxes()

        # Hidden Group
        hidden_group = QGroupBox("Gizli Katman Yapılandırması")
        hidden_layout = QVBoxLayout()
        hidden_count_layout = QHBoxLayout()
        hidden_count_layout.addWidget(QLabel("Gizli Katman Sayısı:"))
        self.hidden_spin.setMinimum(0); self.hidden_spin.setValue(1)
        self.hidden_spin.valueChanged.connect(self.update_hidden_inputs)
        hidden_count_layout.addWidget(self.hidden_spin)
        hidden_layout.addLayout(hidden_count_layout)
        self.hidden_grid = QGridLayout(); hidden_layout.addLayout(self.hidden_grid)
        hidden_group.setLayout(hidden_layout); main_layout.addWidget(hidden_group)
        self.update_hidden_inputs()

        # --- Eğitim Parametreleri (Aktivasyon, Kayıp, Öğrenme Oranı) ---
        train_params_group = QGroupBox("Eğitim Parametreleri")
        train_params_layout = QGridLayout() # Grid layout kullanalım
        train_params_layout.addWidget(QLabel("Gizli Katman Aktivasyonu:"), 0, 0)
        self.activation_combo.addItems(["relu", "sigmoid"])
        train_params_layout.addWidget(self.activation_combo, 0, 1)

        train_params_layout.addWidget(QLabel("Kayıp Fonksiyonu:"), 1, 0)
        self.loss_combo.addItems(["mse", "mae", "rmse", "cross_entropy"])
        train_params_layout.addWidget(self.loss_combo, 1, 1)

        # YENİ: Öğrenme Oranı
        train_params_layout.addWidget(QLabel("Öğrenme Oranı:"), 2, 0)
        self.lr_spinbox.setRange(0.0001, 1.0)
        self.lr_spinbox.setDecimals(4)
        self.lr_spinbox.setSingleStep(0.001)
        self.lr_spinbox.setValue(0.05) # Varsayılan değer
        train_params_layout.addWidget(self.lr_spinbox, 2, 1)
        train_params_group.setLayout(train_params_layout)
        main_layout.addWidget(train_params_group)

        # --- Kontrol Butonları ---
        control_layout = QVBoxLayout() # Butonları dikey gruplayalım

        # YENİ: Çoklu Adım Eğitme Alanı
        multi_step_layout = QHBoxLayout()
        multi_step_layout.addWidget(QLabel("Adım Sayısı:"))
        self.steps_spinbox.setRange(1, 100000) # Geniş bir aralık
        self.steps_spinbox.setValue(100) # Varsayılan adım sayısı
        self.steps_spinbox.setFixedWidth(100)
        multi_step_layout.addWidget(self.steps_spinbox)
        train_multi_btn = QPushButton("Eğit (Adım Sayısı Kadar)")
        train_multi_btn.clicked.connect(self.run_multi_steps) # Yeni metoda bağla
        multi_step_layout.addWidget(train_multi_btn)
        multi_step_layout.addStretch() # Butonları sola yasla
        control_layout.addLayout(multi_step_layout)

        # Mevcut Butonlar (Tek Adım, Parametre Girişi, Görselleştirme)
        single_button_layout = QHBoxLayout()
        run_btn = QPushButton("Adım At (1 Kez Eğit)") # İsmi netleştirelim
        run_btn.clicked.connect(self.run_step)
        single_button_layout.addWidget(run_btn)

        param_btn = QPushButton("Ağırlık/Bias Parametre Girişi")
        param_btn.clicked.connect(self.enter_parameters)
        single_button_layout.addWidget(param_btn)

        diagram_btn = QPushButton("Ağ Yapısını Görselleştir")
        diagram_btn.clicked.connect(self.visualize_architecture)
        single_button_layout.addWidget(diagram_btn)
        single_button_layout.addStretch()
        control_layout.addLayout(single_button_layout)

        main_layout.addLayout(control_layout) # Kontrol butonları grubunu ekle


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

    # --- UI Güncelleme Fonksiyonları (Aynı Kalabilir) ---
    def clear_grid_layout(self, grid_layout):
        while grid_layout.count():
            item = grid_layout.takeAt(0); widget = item.widget()
            if widget: widget.deleteLater()

    def update_input_boxes(self):
        self.clear_grid_layout(self.input_grid); self.input_boxes = []
        cols = 4
        for i in range(self.input_spin.value()):
            label = QLabel(f"x{i+1}:"); edit = QLineEdit("0.0"); edit.setFixedWidth(60)
            row, col = i // cols, (i % cols) * 2
            self.input_grid.addWidget(label, row, col); self.input_grid.addWidget(edit, row, col + 1)
            self.input_boxes.append(edit)
        self.dnn = None # Yapı değişti, sıfırla

    def update_output_boxes(self):
        self.clear_grid_layout(self.output_grid); self.output_boxes = []
        cols = 4
        for i in range(self.output_spin.value()):
            label = QLabel(f"Hedef y{i+1}:"); edit = QLineEdit("1.0"); edit.setFixedWidth(60)
            row, col = i // cols, (i % cols) * 2
            self.output_grid.addWidget(label, row, col); self.output_grid.addWidget(edit, row, col + 1)
            self.output_boxes.append(edit)
        self.dnn = None # Yapı değişti, sıfırla

    def update_hidden_inputs(self):
        self.clear_grid_layout(self.hidden_grid); self.hidden_layer_inputs = []
        cols = 2
        for i in range(self.hidden_spin.value()):
            label = QLabel(f"Gizli Katman {i+1} Nöron Sayısı:"); spin = QSpinBox()
            spin.setMinimum(1); spin.setValue(4); spin.setFixedWidth(80)
            row, col = i // cols, (i % cols) * 2
            self.hidden_grid.addWidget(label, row, col); self.hidden_grid.addWidget(spin, row, col + 1)
            self.hidden_layer_inputs.append(spin)
        self.dnn = None # Yapı değişti, sıfırla

    # --- Yardımcı Metod: DNN Hazırlama ve Veri Alma ---
    def _prepare_dnn_and_data(self):
        """DNN nesnesini ve eğitim verisini UI'dan alıp hazırlar."""
        try:
            # Ağ yapılandırmasını al
            num_inputs = self.input_spin.value()
            num_outputs = self.output_spin.value()
            hidden_nodes = [spin.value() for spin in self.hidden_layer_inputs]
            current_config = [num_inputs] + hidden_nodes + [num_outputs]

            # Giriş (X) ve Hedef (Y) değerlerini al ve doğrula
            x_vals = []
            for box in self.input_boxes:
                try: x_vals.append(float(box.text()))
                except ValueError:
                    QMessageBox.warning(self, "Giriş Hatası", f"Geçersiz giriş değeri: '{box.text()}'."); return False
            self.current_x = np.array(x_vals).reshape(-1, 1)

            y_vals = []
            for box in self.output_boxes:
                 try: y_vals.append(float(box.text()))
                 except ValueError:
                    QMessageBox.warning(self, "Hedef Çıkış Hatası", f"Geçersiz hedef çıkış değeri: '{box.text()}'."); return False
            self.current_y = np.array(y_vals).reshape(-1, 1)

            # Girdi/çıktı sayısı kontrolü
            if len(x_vals) != num_inputs:
                QMessageBox.warning(self, "Yapılandırma Hatası", f"Giriş sayısı ({num_inputs}) ile girilen değer sayısı ({len(x_vals)}) uyuşmuyor."); return False
            if len(y_vals) != num_outputs:
                 QMessageBox.warning(self, "Yapılandırma Hatası", f"Çıkış sayısı ({num_outputs}) ile girilen hedef değer sayısı ({len(y_vals)}) uyuşmuyor."); return False

            # Eğitim parametrelerini al
            activation = self.activation_combo.currentText()
            loss_type = self.loss_combo.currentText()
            learning_rate = self.lr_spinbox.value() # YENİ: Öğrenme oranını al

            # DNN nesnesini kontrol et veya oluştur/güncelle
            config_changed = (self.dnn is None or
                              self.dnn.layer_sizes != current_config or
                              self.dnn.hidden_activation != activation or
                              self.dnn.loss_type != loss_type or
                              self.dnn.learning_rate != learning_rate)

            if config_changed:
                 # Parametre girişiyle oluşturulmuşsa bile, eğitim parametreleri değiştiyse yeniden başlatalım
                 self.dnn = DeepDNN(current_config,
                                    learning_rate=learning_rate, # UI'dan alınan LR
                                    hidden_activation=activation,
                                    loss_type=loss_type)
                 self.result_text.append(f"DNN modeli UI ayarlarına göre oluşturuldu/güncellendi (LR={learning_rate:.4f}).")
                 self.network_config = current_config # Sakla

            return True # Hazırlık başarılı

        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Hazırlık Hatası", f"DNN veya veri hazırlanırken hata: {str(e)}")
            return False


    # --- Eğitim Adımı Çalıştırma Metodları ---
    def run_step(self):
        """Tek bir eğitim adımı çalıştırır."""
        if self._prepare_dnn_and_data(): # Önce hazırlığı yap
            try:
                loss, y_pred = self.dnn.train_step(self.current_x, self.current_y)

                # Sonuçları göster
                self.result_text.append("-" * 20)
                self.result_text.append(f"[1 Adım] Giriş (X): {self.current_x.ravel()}")
                self.result_text.append(f"Hedef (Y): {self.current_y.ravel()}")
                self.result_text.append(f"Tahmin (Y_pred): {np.round(y_pred.ravel(), 4)}")
                self.result_text.append(f"Kayıp ({self.dnn.loss_type}): {loss:.6f}")
                self.result_text.append("")
                self.result_text.verticalScrollBar().setValue(self.result_text.verticalScrollBar().maximum())

            except Exception as e:
                traceback.print_exc()
                QMessageBox.critical(self, "Çalıştırma Hatası", f"Tek adım atılırken bir hata oluştu: {str(e)}")

    # YENİ: Çoklu Adım Eğitme Metodu
    def run_multi_steps(self):
        """Belirtilen adım sayısı kadar eğitim çalıştırır."""
        if self._prepare_dnn_and_data(): # Önce hazırlığı yap
            try:
                num_steps = self.steps_spinbox.value()
                initial_loss = -1
                final_loss = -1
                final_y_pred = None

                self.result_text.append(f"--- {num_steps} Adımlık Eğitim Başlatılıyor ---")
                QApplication.processEvents() # Arayüzün yanıt vermesini sağla

                for i in range(num_steps):
                    loss, y_pred = self.dnn.train_step(self.current_x, self.current_y)
                    if i == 0:
                        initial_loss = loss
                    if i == num_steps - 1: # Son adımın sonuçlarını sakla
                        final_loss = loss
                        final_y_pred = y_pred

                    # İsteğe bağlı: Her N adımda bir loglama yapılabilir
                    # if (i + 1) % (num_steps // 10) == 0: # %10 ilerlemede logla
                    #    self.result_text.append(f"Adım {i+1}/{num_steps}, Kayıp: {loss:.6f}")
                    #    QApplication.processEvents()

                # Sonuçları göster
                self.result_text.append(f"--- {num_steps} Adımlık Eğitim Tamamlandı ---")
                self.result_text.append(f"Giriş (X): {self.current_x.ravel()}")
                self.result_text.append(f"Hedef (Y): {self.current_y.ravel()}")
                if final_y_pred is not None:
                    self.result_text.append(f"Son Tahmin (Y_pred): {np.round(final_y_pred.ravel(), 4)}")
                self.result_text.append(f"Başlangıç Kaybı: {initial_loss:.6f}")
                self.result_text.append(f"Son Kayıp ({self.dnn.loss_type}): {final_loss:.6f}")
                self.result_text.append("")
                self.result_text.verticalScrollBar().setValue(self.result_text.verticalScrollBar().maximum())

            except Exception as e:
                traceback.print_exc()
                QMessageBox.critical(self, "Çalıştırma Hatası", f"{num_steps} adım atılırken bir hata oluştu: {str(e)}")


    # --- Parametre Girişi ve Görselleştirme (Aynı Kalabilir) ---
    def enter_parameters(self):
        """Her bir ağırlık ve bias için ayrı giriş alanı sunan diyalog açar."""
        try:
            num_inputs = self.input_spin.value()
            num_outputs = self.output_spin.value()
            hidden_nodes = [spin.value() for spin in self.hidden_layer_inputs]
            layer_sizes = [num_inputs] + hidden_nodes + [num_outputs]

            if len(layer_sizes) < 2:
                 QMessageBox.warning(self, "Hata", "Parametre girmek için geçerli bir ağ yapısı tanımlanmalı."); return

            dialog = QDialog(self); dialog.setWindowTitle("Parametre Girişi (Tek Tek)")
            dialog.setMinimumWidth(600); dialog.setMinimumHeight(400)
            scroll_area = QScrollArea(dialog); scroll_area.setWidgetResizable(True)
            scroll_content = QWidget(); params_layout = QVBoxLayout(scroll_content)
            self.manual_param_widgets = {}

            for l in range(len(layer_sizes) - 1):
                layer_index = l + 1; prev_layer_size = layer_sizes[l]; current_layer_size = layer_sizes[l+1]
                layer_group = QGroupBox(f"Katman {layer_index} (L{l} -> L{layer_index})"); layer_grid = QGridLayout()
                w_key = f"W{layer_index}"; self.manual_param_widgets[w_key] = []
                layer_grid.addWidget(QLabel(f"<b>Ağırlıklar (W{layer_index})</b> [Hedef <- Kaynak]:"), 0, 0, 1, 4)
                row_offset = 1; col_width = 2
                for j in range(current_layer_size):
                    w_row_widgets = []
                    for k in range(prev_layer_size):
                        label = QLabel(f"W[{j+1}<-{k+1}]:"); spinbox = QDoubleSpinBox()
                        spinbox.setRange(-100.0, 100.0); spinbox.setDecimals(4); spinbox.setSingleStep(0.01); spinbox.setFixedWidth(100)
                        default_val = 0.1
                        if self.dnn and w_key in self.dnn.parameters and j < self.dnn.parameters[w_key].shape[0] and k < self.dnn.parameters[w_key].shape[1]:
                             default_val = self.dnn.parameters[w_key][j, k]
                        spinbox.setValue(default_val)
                        grid_row, grid_col = row_offset + j, k * col_width
                        layer_grid.addWidget(label, grid_row, grid_col); layer_grid.addWidget(spinbox, grid_row, grid_col + 1)
                        w_row_widgets.append(spinbox)
                    self.manual_param_widgets[w_key].append(w_row_widgets)

                b_key = f"b{layer_index}"; self.manual_param_widgets[b_key] = []
                bias_row_start = row_offset + current_layer_size + 1
                layer_grid.addWidget(QLabel(f"<b>Biaslar (b{layer_index})</b> [Nöron]:"), bias_row_start -1 , 0, 1, 4)
                for j in range(current_layer_size):
                    label = QLabel(f"b[{j+1}]:"); spinbox = QDoubleSpinBox()
                    spinbox.setRange(-100.0, 100.0); spinbox.setDecimals(4); spinbox.setSingleStep(0.01); spinbox.setFixedWidth(100)
                    default_val = 0.0
                    if self.dnn and b_key in self.dnn.parameters and j < self.dnn.parameters[b_key].shape[0]:
                       default_val = self.dnn.parameters[b_key][j, 0]
                    spinbox.setValue(default_val)
                    grid_row = bias_row_start + j
                    layer_grid.addWidget(label, grid_row, 0); layer_grid.addWidget(spinbox, grid_row, 1)
                    self.manual_param_widgets[b_key].append(spinbox)
                layer_group.setLayout(layer_grid); params_layout.addWidget(layer_group)

            scroll_area.setWidget(scroll_content)
            buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            buttons.accepted.connect(dialog.accept); buttons.rejected.connect(dialog.reject)
            dialog_layout = QVBoxLayout(dialog); dialog_layout.addWidget(scroll_area); dialog_layout.addWidget(buttons)

            if dialog.exec_() == QDialog.Accepted:
                new_params = {}; learning_rate = self.lr_spinbox.value() # LR'ı da alalım
                try:
                    for l in range(len(layer_sizes) - 1):
                        layer_idx = l + 1; w_key = f"W{layer_idx}"; b_key = f"b{layer_idx}"
                        rows, cols = layer_sizes[layer_idx], layer_sizes[l]
                        W = np.zeros((rows, cols)); b = np.zeros((rows, 1))
                        for j in range(rows):
                            for k in range(cols): W[j, k] = self.manual_param_widgets[w_key][j][k].value()
                        for j in range(rows): b[j, 0] = self.manual_param_widgets[b_key][j].value()
                        new_params[w_key] = W; new_params[b_key] = b

                    if self.dnn is None: # Eğer yoksa yeni oluştur
                         self.dnn = DeepDNN(layer_sizes, learning_rate=learning_rate,
                                            hidden_activation=self.activation_combo.currentText(),
                                            loss_type=self.loss_combo.currentText())
                         self.result_text.append("Yeni DNN modeli oluşturuldu (parametre girişi sonrası).")
                    else: # Varsa öğrenme oranını da güncelle
                         self.dnn.learning_rate = learning_rate

                    self.dnn.parameters = new_params; self.dnn.layer_sizes = layer_sizes
                    QMessageBox.information(self, "Başarılı", "Girilen ağırlık ve bias değerleri ağa yüklendi.")
                    self.result_text.append(f"Manuel parametreler ağa yüklendi (LR={learning_rate:.4f}).")
                    self.manual_param_widgets = {}
                except Exception as parse_error:
                     traceback.print_exc(); QMessageBox.critical(self, "Parametre Yükleme Hatası", f"Hata: {parse_error}")
            else: self.manual_param_widgets = {}
        except Exception as e:
            traceback.print_exc(); QMessageBox.warning(self, "Parametre Girişi Hatası", f"Diyalog hatası: {str(e)}")


    def visualize_architecture(self):
        """Ağ mimarisini, ağırlık ve biasları görselleştirir."""
        try:
            params = None; layer_sizes = []
            if self.dnn is None:
                QMessageBox.information(self, "Bilgi","Ağırlık/Bias görmek için ağı başlatın (Adım At veya Parametre Gir).");
                layer_sizes = [self.input_spin.value()] + [spin.value() for spin in self.hidden_layer_inputs] + [self.output_spin.value()]
                params = None
            else:
                layer_sizes = self.dnn.layer_sizes; params = self.dnn.parameters

            if not layer_sizes or len(layer_sizes) < 2:
                 QMessageBox.warning(self, "Hata", "Görselleştirilecek yapı yok."); return

            max_neurons = max(layer_sizes) if layer_sizes else 1; num_layers_viz = len(layer_sizes)
            fig_width = max(12, num_layers_viz * 3.5); fig_height = max(8, max_neurons * 1.0)
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            neuron_positions = []; node_radius = 0.4; h_spacing = 4.0; v_spacing = 1.5

            for i, layer_size in enumerate(layer_sizes):
                total_height = (layer_size - 1) * v_spacing; y_start = (max_neurons * v_spacing - total_height) / 2.0
                layer_positions = []
                layer_name = f"Giriş (L0)" if i==0 else f"Çıkış (L{i})" if i==num_layers_viz-1 else f"Gizli {i} (L{i})"
                ax.text(i * h_spacing, y_start + total_height + v_spacing, f"{layer_name}\n({layer_size} nöron)",
                        ha='center', va='bottom', fontsize=9, weight='bold')
                for j in range(layer_size):
                    x = i * h_spacing; y = y_start + j * v_spacing
                    circle = plt.Circle((x, y), node_radius, color='skyblue', zorder=4, ec='black')
                    ax.add_patch(circle); ax.text(x, y, f"N{j+1}", ha='center', va='center', zorder=5, fontsize=7)
                    layer_positions.append({'pos': (x, y), 'id': j})
                    if params and i > 0: # Biaslar
                        bias_key = f"b{i}"
                        if bias_key in params and params[bias_key].shape[0] > j:
                            bias_val = params[bias_key][j, 0]
                            ax.text(x, y - node_radius * 1.5, f"b={bias_val:.2f}",ha='center', va='top', zorder=5, fontsize=6, color='red')
                neuron_positions.append(layer_positions)

            if params: # Ağırlıklar
                for l in range(num_layers_viz - 1):
                    weight_key = f"W{l+1}"
                    if weight_key in params:
                        W = params[weight_key]
                        for src_neuron in neuron_positions[l]:
                            for dst_neuron in neuron_positions[l + 1]:
                                src_id, dst_id = src_neuron['id'], dst_neuron['id']
                                if dst_id < W.shape[0] and src_id < W.shape[1]:
                                    src_pos, dst_pos = src_neuron['pos'], dst_neuron['pos']
                                    weight_val = W[dst_id, src_id]
                                    arrow = FancyArrowPatch(src_pos, dst_pos, arrowstyle='->', mutation_scale=10, color='gray', lw=0.5, zorder=1, shrinkA=node_radius*1.2, shrinkB=node_radius*1.2)
                                    ax.add_patch(arrow)
                                    mid_x, mid_y = src_pos[0]*0.6+dst_pos[0]*0.4, src_pos[1]*0.6+dst_pos[1]*0.4
                                    ax.text(mid_x, mid_y, f"w={weight_val:.2f}", fontsize=6, color='blue', ha='center', va='center', zorder=3, bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.7, ec='none'))
                                # else: print(f"Warning: Index out of bounds for W{l+1}") # İsteğe bağlı uyarı
            else: # Sadece oklar
                 for l in range(num_layers_viz - 1):
                     for src_neuron in neuron_positions[l]:
                         for dst_neuron in neuron_positions[l + 1]:
                             arrow = FancyArrowPatch(src_neuron['pos'], dst_neuron['pos'], arrowstyle='->', mutation_scale=10, color='lightgray', lw=0.5, zorder=1, shrinkA=node_radius*1.2, shrinkB=node_radius*1.2)
                             ax.add_patch(arrow)

            ax.set_xlim(-node_radius*2, (num_layers_viz - 1) * h_spacing + node_radius*2)
            all_y = [n['pos'][1] for layer in neuron_positions for n in layer] if neuron_positions else [0]
            min_y_coord, max_y_coord = min(all_y) - node_radius*3, max(all_y) + node_radius*2 + v_spacing
            if max_y_coord - min_y_coord < 2: min_y_coord -= 1; max_y_coord += 1
            ax.set_ylim(min_y_coord, max_y_coord); ax.axis('off')
            plt.title("DNN Mimarisi - Ağırlıklar ve Biaslar (varsa)", fontsize=12)
            plt.tight_layout(pad=2.0); plt.show() # GUI'yi kilitler
        except Exception as e:
            traceback.print_exc(); QMessageBox.warning(self, "Görselleştirme Hatası", f"Hata: {str(e)}")


# Ana uygulama kısmı
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DNNConfigurator()
    window.show()
    sys.exit(app.exec_())
