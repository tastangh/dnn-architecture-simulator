from PyQt5.QtWidgets import (
    QWidget, QLabel, QSpinBox, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QGridLayout, QTextEdit, QMessageBox, QGroupBox,
    QScrollArea, QDialog, QFormLayout, QDialogButtonBox, QComboBox,
    QDoubleSpinBox, QApplication, QSizePolicy
)

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch

from backend import DeepDNN
import numpy as np
import sys
import traceback 

class DNNConfigurator(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DNN Architecture Simulator")
        self.resize(1200, 850) 

        # Arayüz Elemanları
        self.input_spin = QSpinBox()
        self.output_spin = QSpinBox()
        self.hidden_spin = QSpinBox()
        self.activation_combo = QComboBox()
        self.loss_combo = QComboBox()
        self.lr_spinbox = QDoubleSpinBox()
        self.steps_spinbox = QSpinBox()

        # Diğer Üyeler
        self.input_boxes = []
        self.hidden_layer_inputs = []
        self.output_boxes = []
        self.manual_param_widgets = {}
        self.result_text = QTextEdit()
        self.network_config = None
        self.dnn = None
        self.current_x = None
        self.current_y = None
        self.loss_history = []
        self.total_steps = 0  

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        # --- Yapılandırma Grupları ---
        config_h_layout = QHBoxLayout() 

        # Sol Taraf: Input, Output, Hidden
        left_config_layout = QVBoxLayout()
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
        input_group.setLayout(input_layout); left_config_layout.addWidget(input_group)
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
        output_group.setLayout(output_layout); left_config_layout.addWidget(output_group)
        self.update_output_boxes()
        left_config_layout.addStretch() 
        config_h_layout.addLayout(left_config_layout)

        # Sağ Taraf: Hidden Layers, Training Params
        right_config_layout = QVBoxLayout()
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
        hidden_group.setLayout(hidden_layout); right_config_layout.addWidget(hidden_group)
        self.update_hidden_inputs()

        # Eğitim Parametreleri Group
        train_params_group = QGroupBox("Eğitim Parametreleri")
        train_params_layout = QGridLayout()
        train_params_layout.addWidget(QLabel("Gizli K. Aktivasyonu:"), 0, 0)
        self.activation_combo.addItems(["relu", "sigmoid"])
        train_params_layout.addWidget(self.activation_combo, 0, 1)
        train_params_layout.addWidget(QLabel("Kayıp Fonksiyonu:"), 1, 0)
        self.loss_combo.addItems(["mse", "mae", "rmse", "cross_entropy"])
        train_params_layout.addWidget(self.loss_combo, 1, 1)
        train_params_layout.addWidget(QLabel("Öğrenme Oranı:"), 2, 0)
        self.lr_spinbox.setRange(0.0001, 1.0); self.lr_spinbox.setDecimals(4)
        self.lr_spinbox.setSingleStep(0.001); self.lr_spinbox.setValue(0.05)
        train_params_layout.addWidget(self.lr_spinbox, 2, 1)
        train_params_group.setLayout(train_params_layout)
        right_config_layout.addWidget(train_params_group)
        right_config_layout.addStretch()
        config_h_layout.addLayout(right_config_layout)

        main_layout.addLayout(config_h_layout) 

        # --- Kontrol Butonları ---
        control_group = QGroupBox("Kontroller")
        control_grid_layout = QGridLayout() 

        # Çoklu Adım Eğitme Alanı
        control_grid_layout.addWidget(QLabel("Adım Sayısı:"), 0, 0)
        self.steps_spinbox.setRange(1, 100000); self.steps_spinbox.setValue(100)
        self.steps_spinbox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed) 
        control_grid_layout.addWidget(self.steps_spinbox, 0, 1)
        train_multi_btn = QPushButton("Eğit (Adım Sayısı Kadar)")
        train_multi_btn.clicked.connect(self.run_multi_steps)
        control_grid_layout.addWidget(train_multi_btn, 0, 2)

        # Tek Adım Butonu
        run_btn = QPushButton("Adım At (1 Kez Eğit)")
        run_btn.clicked.connect(self.run_step)
        control_grid_layout.addWidget(run_btn, 1, 0)

        # Parametre Girişi Butonu
        param_btn = QPushButton("Ağırlık/Bias Parametre Girişi")
        param_btn.clicked.connect(self.enter_parameters)
        control_grid_layout.addWidget(param_btn, 1, 1)

        # Görselleştirme Butonu
        diagram_btn = QPushButton("Ağ Yapısını Görselleştir")
        diagram_btn.clicked.connect(self.visualize_architecture)
        control_grid_layout.addWidget(diagram_btn, 1, 2)

        # YENİ: Kayıp Grafiği Butonu
        loss_plot_btn = QPushButton("Kayıp Grafiğini Çizdir")
        loss_plot_btn.clicked.connect(self.plot_loss_graph)
        control_grid_layout.addWidget(loss_plot_btn, 2, 0)

        # YENİ: Sıfırla Butonu
        reset_btn = QPushButton("Ağı Sıfırla")
        reset_btn.clicked.connect(self.reset_network)
        control_grid_layout.addWidget(reset_btn, 2, 1)

        control_group.setLayout(control_grid_layout)
        main_layout.addWidget(control_group) 


        # --- Sonuçlar Alanı ---
        result_group = QGroupBox("Sonuçlar")
        result_layout = QVBoxLayout()
        self.result_text.setReadOnly(True)
        self.result_text.setMinimumHeight(120) 
        self.result_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding) 
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

    # --- UI Güncelleme Fonksiyonları ---
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
        if self.dnn: # Yapı değiştiğinde sıfırla
            self.reset_network(show_message=False) # Mesaj göstermeden sıfırla


    def update_output_boxes(self):
        self.clear_grid_layout(self.output_grid); self.output_boxes = []
        cols = 4
        for i in range(self.output_spin.value()):
            label = QLabel(f"Hedef y{i+1}:"); edit = QLineEdit("1.0"); edit.setFixedWidth(60)
            row, col = i // cols, (i % cols) * 2
            self.output_grid.addWidget(label, row, col); self.output_grid.addWidget(edit, row, col + 1)
            self.output_boxes.append(edit)
        if self.dnn:
            self.reset_network(show_message=False)


    def update_hidden_inputs(self):
        self.clear_grid_layout(self.hidden_grid); self.hidden_layer_inputs = []
        cols = 2
        for i in range(self.hidden_spin.value()):
            label = QLabel(f"Gizli K. {i+1} Nöron:"); spin = QSpinBox()
            spin.setMinimum(1); spin.setValue(4); spin.setFixedWidth(80)
            row, col = i // cols, (i % cols) * 2
            self.hidden_grid.addWidget(label, row, col); self.hidden_grid.addWidget(spin, row, col + 1)
            self.hidden_layer_inputs.append(spin)
        if self.dnn:
             self.reset_network(show_message=False)


    # --- Yardımcı Metod: DNN Hazırlama ve Veri Alma ---
    def _prepare_dnn_and_data(self):
        """DNN nesnesini ve eğitim verisini UI'dan alıp hazırlar."""
        try:
            num_inputs = self.input_spin.value()
            num_outputs = self.output_spin.value()
            hidden_nodes = [spin.value() for spin in self.hidden_layer_inputs]
            current_config = [num_inputs] + hidden_nodes + [num_outputs]

            x_vals = []
            for box in self.input_boxes:
                try: x_vals.append(float(box.text()))
                except ValueError: QMessageBox.warning(self, "Giriş Hatası", f"Geçersiz: '{box.text()}'."); return False
            self.current_x = np.array(x_vals).reshape(-1, 1)

            y_vals = []
            for box in self.output_boxes:
                 try: y_vals.append(float(box.text()))
                 except ValueError: QMessageBox.warning(self, "Hedef Çıkış Hatası", f"Geçersiz: '{box.text()}'."); return False
            self.current_y = np.array(y_vals).reshape(-1, 1)

            if len(x_vals) != num_inputs: QMessageBox.warning(self, "Yapılandırma Hatası", f"Giriş sayısı/değer sayısı uyuşmuyor."); return False
            if len(y_vals) != num_outputs: QMessageBox.warning(self, "Yapılandırma Hatası", f"Çıkış sayısı/hedef sayısı uyuşmuyor."); return False

            activation = self.activation_combo.currentText()
            loss_type = self.loss_combo.currentText()
            learning_rate = self.lr_spinbox.value()

            # Yapı veya ana hiperparametreler değiştiyse DNN'i sıfırdan oluştur
            config_changed = (self.dnn is None or
                              self.dnn.layer_sizes != current_config or
                              self.dnn.hidden_activation != activation or
                              self.dnn.loss_type != loss_type) # LR değişikliği parametreleri etkilemez, ama yine de başlatabiliriz

            if config_changed:
                 self.dnn = DeepDNN(current_config, learning_rate=learning_rate,
                                    hidden_activation=activation, loss_type=loss_type)
                 self.result_text.append(f"DNN modeli UI ayarlarına göre oluşturuldu/güncellendi (LR={learning_rate:.4f}).")
                 self.network_config = current_config
                 # Yeni yapılandırma için geçmişi sıfırla
                 self.loss_history = []
                 self.total_steps = 0
            else:
                # Sadece öğrenme oranı değiştiyse, mevcut DNN'de güncelle
                 if self.dnn.learning_rate != learning_rate:
                     self.dnn.learning_rate = learning_rate
                     self.result_text.append(f"Öğrenme oranı {learning_rate:.4f} olarak güncellendi.")

            return True

        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Hazırlık Hatası", f"Hata: {str(e)}")
            return False


    # --- Eğitim Adımı Çalıştırma Metodları ---
    def run_step(self):
        """Tek bir eğitim adımı çalıştırır."""
        if self._prepare_dnn_and_data():
            try:
                loss, y_pred = self.dnn.train_step(self.current_x, self.current_y)

                # Adım ve Kayıp Geçmişini Güncelle
                self.total_steps += 1
                self.loss_history.append((self.total_steps, loss))

                # Sonuçları göster
                self.result_text.append("-" * 20)
                self.result_text.append(f"[Adım {self.total_steps}] Giriş (X): {self.current_x.ravel()}")
                self.result_text.append(f"Hedef (Y): {self.current_y.ravel()}")
                self.result_text.append(f"Tahmin (Y_pred): {np.round(y_pred.ravel(), 4)}")
                self.result_text.append(f"Kayıp ({self.dnn.loss_type}): {loss:.6f}")
                self.result_text.append("")
                self.result_text.verticalScrollBar().setValue(self.result_text.verticalScrollBar().maximum())

            except Exception as e:
                traceback.print_exc()
                QMessageBox.critical(self, "Çalıştırma Hatası", f"Tek adım atılırken hata: {str(e)}")

    def run_multi_steps(self):
        """Belirtilen adım sayısı kadar eğitim çalıştırır."""
        if self._prepare_dnn_and_data():
            try:
                num_steps_to_run = self.steps_spinbox.value()
                initial_loss = -1
                final_loss = -1
                final_y_pred = None
                start_step = self.total_steps + 1

                self.result_text.append(f"--- {num_steps_to_run} Adımlık Eğitim Başlatılıyor (Adım {start_step}'den itibaren) ---")
                QApplication.processEvents()

                for i in range(num_steps_to_run):
                    current_step_number = self.total_steps + 1
                    loss, y_pred = self.dnn.train_step(self.current_x, self.current_y)

                    # Adım ve Kayıp Geçmişini Güncelle
                    self.total_steps = current_step_number
                    self.loss_history.append((self.total_steps, loss))

                    if i == 0: initial_loss = loss # Döngünün ilk kaybı
                    if i == num_steps_to_run - 1: # Son adımın sonuçları
                        final_loss = loss
                        final_y_pred = y_pred

                    # İsteğe Bağlı Loglama (her 10% veya 100 adımda bir)
                    log_interval = max(1, num_steps_to_run // 10) if num_steps_to_run > 100 else 100
                    if (i + 1) % log_interval == 0 and num_steps_to_run > 1:
                       self.result_text.append(f"  Adım {self.total_steps}/{start_step + num_steps_to_run - 1}, Kayıp: {loss:.6f}")
                       QApplication.processEvents()


                # Sonuçları göster
                self.result_text.append(f"--- {num_steps_to_run} Adımlık Eğitim Tamamlandı (Son Adım: {self.total_steps}) ---")
                self.result_text.append(f"Giriş (X): {self.current_x.ravel()}") # Sabit giriş varsayımı
                self.result_text.append(f"Hedef (Y): {self.current_y.ravel()}") # Sabit hedef varsayımı
                if final_y_pred is not None:
                    self.result_text.append(f"Son Tahmin (Y_pred): {np.round(final_y_pred.ravel(), 4)}")
                self.result_text.append(f"Başlangıç Kaybı (Adım {start_step}): {initial_loss:.6f}")
                self.result_text.append(f"Son Kayıp (Adım {self.total_steps}): {final_loss:.6f}")
                self.result_text.append("")
                self.result_text.verticalScrollBar().setValue(self.result_text.verticalScrollBar().maximum())

            except Exception as e:
                traceback.print_exc()
                QMessageBox.critical(self, "Çalıştırma Hatası", f"{num_steps_to_run} adım atılırken hata: {str(e)}")


    # --- Parametre Girişi ---
    def enter_parameters(self):
        """Her bir ağırlık ve bias için ayrı giriş alanı sunan diyalog açar."""
        try:
            num_inputs = self.input_spin.value()
            num_outputs = self.output_spin.value()
            hidden_nodes = [spin.value() for spin in self.hidden_layer_inputs]
            layer_sizes = [num_inputs] + hidden_nodes + [num_outputs]

            if len(layer_sizes) < 2: QMessageBox.warning(self, "Hata", "Parametre girmek için geçerli yapı yok."); return

            dialog = QDialog(self); dialog.setWindowTitle("Parametre Girişi (Tek Tek)")
            dialog.setMinimumWidth(600); dialog.setMinimumHeight(400)
            scroll_area = QScrollArea(dialog); scroll_area.setWidgetResizable(True)
            scroll_content = QWidget(); params_layout = QVBoxLayout(scroll_content)
            self.manual_param_widgets = {} # Her diyalogda sıfırla

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
                new_params = {}; learning_rate = self.lr_spinbox.value()
                try:
                    for l in range(len(layer_sizes) - 1):
                        layer_idx = l + 1; w_key = f"W{layer_idx}"; b_key = f"b{layer_idx}"
                        rows, cols = layer_sizes[layer_idx], layer_sizes[l]
                        W = np.zeros((rows, cols)); b = np.zeros((rows, 1))
                        for j in range(rows):
                            for k in range(cols): W[j, k] = self.manual_param_widgets[w_key][j][k].value()
                        for j in range(rows): b[j, 0] = self.manual_param_widgets[b_key][j].value()
                        new_params[w_key] = W; new_params[b_key] = b

                    if self.dnn is None:
                         self.dnn = DeepDNN(layer_sizes, learning_rate=learning_rate,
                                            hidden_activation=self.activation_combo.currentText(),
                                            loss_type=self.loss_combo.currentText())
                         self.result_text.append("Yeni DNN modeli (manuel parametrelerle) oluşturuldu.")
                    else:
                         self.dnn.learning_rate = learning_rate

                    self.dnn.parameters = new_params; self.dnn.layer_sizes = layer_sizes
                    # Manuel parametre girişi yapıldığında eğitim geçmişini sıfırla
                    self.loss_history = []
                    self.total_steps = 0
                    QMessageBox.information(self, "Başarılı", "Parametreler ağa yüklendi.")
                    self.result_text.append(f"Manuel parametreler ağa yüklendi (LR={learning_rate:.4f}). Adım sayacı sıfırlandı.")
                    self.manual_param_widgets = {} # Temizle
                except Exception as parse_error:
                     traceback.print_exc(); QMessageBox.critical(self, "Parametre Yükleme Hatası", f"Hata: {parse_error}")
            else: self.manual_param_widgets = {} # İptal edilirse de temizle
        except Exception as e:
            traceback.print_exc(); QMessageBox.warning(self, "Parametre Girişi Hatası", f"Diyalog hatası: {str(e)}")


    # --- Görselleştirme ve Sıfırlama ---
    def visualize_architecture(self):
        """Ağ mimarisini, ağırlık ve biasları görselleştirir."""
        try:
            params = None; layer_sizes = []
            if self.dnn is None:
                QMessageBox.information(self, "Bilgi","Ağırlık/Bias görmek için ağı başlatın.");
                layer_sizes = [self.input_spin.value()] + [spin.value() for spin in self.hidden_layer_inputs] + [self.output_spin.value()]
                params = None
            else:
                layer_sizes = self.dnn.layer_sizes; params = self.dnn.parameters

            if not layer_sizes or len(layer_sizes) < 2: QMessageBox.warning(self, "Hata", "Görselleştirilecek yapı yok."); return

            max_neurons = max(layer_sizes) if layer_sizes else 1; num_layers_viz = len(layer_sizes)
            fig_width = max(12, num_layers_viz * 3.5); fig_height = max(8, max_neurons * 1.0)
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            neuron_positions = []; node_radius = 0.4; h_spacing = 4.0; v_spacing = 1.5

            for i, layer_size in enumerate(layer_sizes):
                total_height = (layer_size - 1) * v_spacing; y_start = (max_neurons * v_spacing - total_height) / 2.0
                layer_positions = []
                layer_name = f"Giriş (L0)" if i==0 else f"Çıkış (L{i})" if i==num_layers_viz-1 else f"Gizli {i} (L{i})"
                ax.text(i * h_spacing, y_start + total_height + v_spacing, f"{layer_name}\n({layer_size} nöron)", ha='center', va='bottom', fontsize=9, weight='bold')
                for j in range(layer_size):
                    x = i * h_spacing; y = y_start + j * v_spacing
                    circle = plt.Circle((x, y), node_radius, color='skyblue', zorder=4, ec='black')
                    ax.add_patch(circle); ax.text(x, y, f"N{j+1}", ha='center', va='center', zorder=5, fontsize=7)
                    layer_positions.append({'pos': (x, y), 'id': j})
                    if params and i > 0: # Biaslar
                        bias_key = f"b{i}"
                        if bias_key in params and params[bias_key].shape[0] > j:
                            bias_val = params[bias_key][j, 0]; ax.text(x, y - node_radius * 1.5, f"b={bias_val:.2f}",ha='center', va='top', zorder=5, fontsize=6, color='red')
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
                                    src_pos, dst_pos = src_neuron['pos'], dst_neuron['pos']; weight_val = W[dst_id, src_id]
                                    arrow = FancyArrowPatch(src_pos, dst_pos, arrowstyle='->', mutation_scale=10, color='gray', lw=0.5, zorder=1, shrinkA=node_radius*1.2, shrinkB=node_radius*1.2)
                                    ax.add_patch(arrow)
                                    mid_x, mid_y = src_pos[0]*0.6+dst_pos[0]*0.4, src_pos[1]*0.6+dst_pos[1]*0.4
                                    ax.text(mid_x, mid_y, f"w={weight_val:.2f}", fontsize=6, color='blue', ha='center', va='center', zorder=3, bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.7, ec='none'))
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

    def plot_loss_graph(self):
        """Eğitim sırasındaki kayıp değerlerini çizer."""
        if not self.loss_history:
            QMessageBox.information(self, "Bilgi", "Gösterilecek kayıp geçmişi bulunmuyor. Lütfen önce ağı eğitin.")
            return

        try:
            steps, losses = zip(*self.loss_history) # Adım ve kayıp değerlerini ayır

            plt.figure(figsize=(10, 6)) # Yeni bir figür oluştur
            plt.plot(steps, losses, marker='.', linestyle='-', markersize=4)
            plt.xlabel("Eğitim Adımı Sayısı")
            plt.ylabel("Kayıp Değeri")
            plt.title(f"Eğitim Kayıp Grafiği ({self.dnn.loss_type if self.dnn else 'N/A'})")
            plt.grid(True)
            plt.tight_layout()
            plt.show() # GUI'yi kilitler

        except Exception as e:
             traceback.print_exc(); QMessageBox.warning(self, "Grafik Hatası", f"Kayıp grafiği çizilirken hata: {str(e)}")


    def reset_network(self, show_message=True):
        """DNN durumunu, sonuçları ve kayıp geçmişini sıfırlar."""
        self.dnn = None
        self.network_config = None
        self.loss_history = []
        self.total_steps = 0
        self.current_x = None
        self.current_y = None
        self.result_text.clear()
        self.result_text.append("Ağ durumu ve sonuçlar sıfırlandı.")

        if show_message:
             QMessageBox.information(self, "Sıfırlandı", "Yapay sinir ağı durumu, sonuçlar ve kayıp geçmişi sıfırlandı.")


# Ana uygulama kısmı (main.py yerine burada çalıştırılabilir)
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DNNConfigurator()
    window.show()
    sys.exit(app.exec_())
