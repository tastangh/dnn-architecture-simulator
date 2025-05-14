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

        self.input_spin = QSpinBox()
        self.output_spin = QSpinBox()
        self.hidden_spin = QSpinBox() # Gizli katman sayısı
        
        self.output_layer_activation_combo = QComboBox() # Çıkış katmanı için ayrı bir tane tutalım

        self.loss_combo = QComboBox()
        self.lr_spinbox = QDoubleSpinBox()
        self.leaky_relu_alpha_spinbox = QDoubleSpinBox()
        self.steps_spinbox = QSpinBox()

        self.input_boxes = []
        self.hidden_layer_configs = [] # [(QSpinBox_neurons, QComboBox_activation), ...]
        self.output_boxes = []
        self.manual_param_widgets = {}
        self.result_text = QTextEdit()
        self.network_config = None # layer_sizes
        self.activations_config = None # activations list
        self.dnn = None
        self.current_x = None
        self.current_y = None
        self.loss_history = []
        self.total_steps = 0

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        config_h_layout = QHBoxLayout()
        left_config_layout = QVBoxLayout()
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

        output_group = QGroupBox("Output Yapılandırması")
        output_layout = QVBoxLayout()
        output_top_layout = QHBoxLayout() # Nöron sayısı ve aktivasyon için
        
        output_count_layout = QHBoxLayout()
        output_count_layout.addWidget(QLabel("Çıkış Sayısı:"))
        self.output_spin.setMinimum(1); self.output_spin.setValue(1)
        self.output_spin.valueChanged.connect(self.update_output_boxes)
        output_count_layout.addWidget(self.output_spin)
        output_top_layout.addLayout(output_count_layout)
        output_top_layout.addStretch(1)

        output_act_layout = QHBoxLayout()
        output_act_layout.addWidget(QLabel("Çıkış Aktivasyonu:"))
        self.output_layer_activation_combo.addItems(["sigmoid", "softmax", "linear", "relu", "tanh", "leaky_relu"])
        self.output_layer_activation_combo.setCurrentText("sigmoid")
        self.output_layer_activation_combo.setFixedWidth(100)
        output_act_layout.addWidget(self.output_layer_activation_combo)
        output_top_layout.addLayout(output_act_layout)
        
        output_layout.addLayout(output_top_layout)
        self.output_grid = QGridLayout(); output_layout.addLayout(self.output_grid) # Hedef değer girişleri
        output_group.setLayout(output_layout); left_config_layout.addWidget(output_group)
        self.update_output_boxes()

        self.output_layer_activation_combo.currentIndexChanged.connect(lambda: self.reset_network(show_message=False))


        left_config_layout.addStretch()
        config_h_layout.addLayout(left_config_layout)

        right_config_layout = QVBoxLayout()

        hidden_group = QGroupBox("Gizli Katman Yapılandırması")
        hidden_layout = QVBoxLayout()
        hidden_count_layout = QHBoxLayout()
        hidden_count_layout.addWidget(QLabel("Gizli Katman Sayısı:"))
        self.hidden_spin.setMinimum(0); self.hidden_spin.setValue(1)
        self.hidden_spin.valueChanged.connect(self.update_hidden_inputs) 
        hidden_count_layout.addWidget(self.hidden_spin)
        hidden_layout.addLayout(hidden_count_layout)
        self.hidden_configs_grid = QGridLayout(); hidden_layout.addLayout(self.hidden_configs_grid)
        hidden_group.setLayout(hidden_layout); right_config_layout.addWidget(hidden_group)
        self.update_hidden_inputs() # Başlangıçta çağır

        # --- Eğitim Parametreleri Group ---
        train_params_group = QGroupBox("Eğitim Parametreleri")
        train_params_layout = QGridLayout()

        train_params_layout.addWidget(QLabel("Kayıp Fonksiyonu:"), 0, 0) # Satır indeksi 0 oldu
        self.loss_combo.addItems(["mse", "mae", "rmse", "cross_entropy"])
        train_params_layout.addWidget(self.loss_combo, 0, 1)
        train_params_layout.addWidget(QLabel("Öğrenme Oranı:"), 1, 0) # Satır indeksi 1 oldu
        self.lr_spinbox.setRange(0.0001, 1.0); self.lr_spinbox.setDecimals(4)
        self.lr_spinbox.setSingleStep(0.001); self.lr_spinbox.setValue(0.05)
        train_params_layout.addWidget(self.lr_spinbox, 1, 1)
        train_params_layout.addWidget(QLabel("Leaky ReLU Alpha:"), 2, 0) # Satır indeksi 2 oldu
        self.leaky_relu_alpha_spinbox.setRange(0.001, 0.5); self.leaky_relu_alpha_spinbox.setDecimals(3)
        self.leaky_relu_alpha_spinbox.setSingleStep(0.001); self.leaky_relu_alpha_spinbox.setValue(0.01)
        train_params_layout.addWidget(self.leaky_relu_alpha_spinbox, 2, 1)
        train_params_group.setLayout(train_params_layout)
        right_config_layout.addWidget(train_params_group)
        right_config_layout.addStretch()
        config_h_layout.addLayout(right_config_layout)
        main_layout.addLayout(config_h_layout)

        control_group = QGroupBox("Kontroller")
        control_grid_layout = QGridLayout()
        control_grid_layout.addWidget(QLabel("Adım Sayısı:"), 0, 0)
        self.steps_spinbox.setRange(1, 100000); self.steps_spinbox.setValue(100)
        self.steps_spinbox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        control_grid_layout.addWidget(self.steps_spinbox, 0, 1)
        train_multi_btn = QPushButton("Eğit (Adım Sayısı Kadar)")
        train_multi_btn.clicked.connect(self.run_multi_steps)
        control_grid_layout.addWidget(train_multi_btn, 0, 2)
        run_btn = QPushButton("Adım At (1 Kez Eğit)")
        run_btn.clicked.connect(self.run_step)
        control_grid_layout.addWidget(run_btn, 1, 0)
        param_btn = QPushButton("Ağırlık/Bias Parametre Girişi")
        param_btn.clicked.connect(self.enter_parameters)
        control_grid_layout.addWidget(param_btn, 1, 1)
        diagram_btn = QPushButton("Ağ Yapısını Görselleştir")
        diagram_btn.clicked.connect(self.visualize_architecture)
        control_grid_layout.addWidget(diagram_btn, 1, 2)
        loss_plot_btn = QPushButton("Kayıp Grafiğini Çizdir")
        loss_plot_btn.clicked.connect(self.plot_loss_graph)
        control_grid_layout.addWidget(loss_plot_btn, 2, 0)
        reset_btn = QPushButton("Ağı Sıfırla")
        reset_btn.clicked.connect(self.reset_network)
        control_grid_layout.addWidget(reset_btn, 2, 1)
        control_group.setLayout(control_grid_layout)
        main_layout.addWidget(control_group)

        result_group = QGroupBox("Sonuçlar")
        result_layout = QVBoxLayout()
        self.result_text.setReadOnly(True)
        self.result_text.setMinimumHeight(120)
        self.result_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        result_layout.addWidget(self.result_text)
        result_group.setLayout(result_layout)
        main_layout.addWidget(result_group)

        container = QWidget()
        container.setLayout(main_layout)
        scroll = QScrollArea()
        scroll.setWidget(container)
        scroll.setWidgetResizable(True)
        window_layout = QVBoxLayout(self)
        window_layout.addWidget(scroll)
        self.setLayout(window_layout)


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
        if self.dnn: self.reset_network(show_message=False)


    def update_output_boxes(self):

        self.clear_grid_layout(self.output_grid); self.output_boxes = []
        cols = 4
        for i in range(self.output_spin.value()):
            label = QLabel(f"Hedef y{i+1}:"); edit = QLineEdit("1.0"); edit.setFixedWidth(60)
            row, col = i // cols, (i % cols) * 2
            self.output_grid.addWidget(label, row, col); self.output_grid.addWidget(edit, row, col + 1)
            self.output_boxes.append(edit)
        if self.dnn: self.reset_network(show_message=False)


    def update_hidden_inputs(self): 
        self.clear_grid_layout(self.hidden_configs_grid)
        self.hidden_layer_configs = [] # Temizle

        num_hidden_layers = self.hidden_spin.value()
        if num_hidden_layers == 0:
            self.hidden_configs_grid.addWidget(QLabel("Gizli katman yok."), 0, 0)

        for i in range(num_hidden_layers):
            # Nöron Sayısı
            neuron_label = QLabel(f"Gizli K. {i+1} Nöron:")
            neuron_spin = QSpinBox()
            neuron_spin.setMinimum(1); neuron_spin.setValue(4); neuron_spin.setFixedWidth(70)
            
            # Aktivasyon Fonksiyonu
            activation_label = QLabel("Aktivasyon:")
            activation_combo = QComboBox()
            activation_combo.addItems(["relu", "sigmoid", "tanh", "leaky_relu", "linear"]) # Softmax genellikle çıkış için
            activation_combo.setCurrentText("relu") # Varsayılan
            activation_combo.setFixedWidth(100)
            # Değişiklik olduğunda ağı sıfırla
            activation_combo.currentIndexChanged.connect(lambda: self.reset_network(show_message=False))
            neuron_spin.valueChanged.connect(lambda: self.reset_network(show_message=False))


            self.hidden_configs_grid.addWidget(neuron_label, i, 0)
            self.hidden_configs_grid.addWidget(neuron_spin, i, 1)
            self.hidden_configs_grid.addWidget(activation_label, i, 2)
            self.hidden_configs_grid.addWidget(activation_combo, i, 3)
            
            self.hidden_layer_configs.append({'neurons': neuron_spin, 'activation': activation_combo})
        
        if self.dnn: # Yapı değiştiğinde sıfırla
             self.reset_network(show_message=False)

    def _prepare_dnn_and_data(self):
        try:
            num_inputs = self.input_spin.value()
            num_outputs = self.output_spin.value()
            
            hidden_nodes = []
            hidden_activations_list = []
            for config in self.hidden_layer_configs:
                hidden_nodes.append(config['neurons'].value())
                hidden_activations_list.append(config['activation'].currentText())

            output_activation_text = self.output_layer_activation_combo.currentText()
            
            current_layer_config = [num_inputs] + hidden_nodes + [num_outputs]
            current_activations_config = hidden_activations_list + [output_activation_text]

            x_vals = []
            for box in self.input_boxes:
                try: x_vals.append(float(box.text()))
                except ValueError: QMessageBox.warning(self, "Giriş Hatası", f"Geçersiz giriş: '{box.text()}'. Lütfen sayısal değer girin."); return False
            self.current_x = np.array(x_vals).reshape(-1, 1)

            y_vals = []
            for box in self.output_boxes:
                 try: y_vals.append(float(box.text()))
                 except ValueError: QMessageBox.warning(self, "Hedef Çıkış Hatası", f"Geçersiz hedef: '{box.text()}'. Lütfen sayısal değer girin."); return False
            self.current_y = np.array(y_vals).reshape(-1, 1)

            if len(x_vals) != num_inputs: QMessageBox.warning(self, "Yapılandırma Hatası", f"Giriş sayısı ({num_inputs}) ile girilen değer sayısı ({len(x_vals)}) uyuşmuyor."); return False
            if len(y_vals) != num_outputs: QMessageBox.warning(self, "Yapılandırma Hatası", f"Çıkış sayısı ({num_outputs}) ile girilen hedef sayısı ({len(y_vals)}) uyuşmuyor."); return False


            loss_type = self.loss_combo.currentText()
            learning_rate = self.lr_spinbox.value()
            leaky_alpha = self.leaky_relu_alpha_spinbox.value()

            # Uyarılar (softmax, cross-entropy)
            if output_activation_text == "softmax" and num_outputs == 1:
                QMessageBox.warning(self, "Yapılandırma Uyarısı",
                                      "Softmax aktivasyonu genellikle 1'den fazla çıkış nöronu için kullanılır. "
                                      "Tek çıkış için sigmoid daha yaygındır.")
            if loss_type == "cross_entropy" and output_activation_text not in ["sigmoid", "softmax"]:
                QMessageBox.warning(self, "Yapılandırma Uyarısı",
                                      f"Cross-entropy kaybı genellikle sigmoid veya softmax çıkış aktivasyonu ile kullanılır. "
                                      f"Seçilen çıkış aktivasyonu ({output_activation_text}) ile sonuçlar beklenmedik olabilir.")

            config_changed = (self.dnn is None or
                              self.dnn.layer_sizes != current_layer_config or
                              self.dnn.activations != current_activations_config or # Aktivasyon listesi kontrolü
                              self.dnn.loss_type != loss_type or
                              (any(act == "leaky_relu" for act in current_activations_config) and self.dnn.leaky_relu_alpha != leaky_alpha) or
                              self.dnn.learning_rate != learning_rate # Sadece LR değişse bile logla
                             )
            
            if config_changed or self.dnn is None:
                 self.dnn = DeepDNN(current_layer_config,
                                    learning_rate=learning_rate,
                                    activations=current_activations_config, # Yeni parametre
                                    loss_type=loss_type,
                                    leaky_relu_alpha=leaky_alpha)
                 self.result_text.append(f"DNN modeli UI ayarlarına göre oluşturuldu/güncellendi (LR={learning_rate:.4f}, Alpha={leaky_alpha:.3f}).")
                 self.network_config = current_layer_config
                 self.activations_config = current_activations_config # Sakla
                 self.loss_history = []
                 self.total_steps = 0
            elif self.dnn.learning_rate != learning_rate:
                 self.dnn.learning_rate = learning_rate
                 self.result_text.append(f"Öğrenme oranı {learning_rate:.4f} olarak güncellendi.")
            # Leaky alpha değişikliği için de benzer bir kontrol eklenebilir
            elif any(act == "leaky_relu" for act in current_activations_config) and self.dnn.leaky_relu_alpha != leaky_alpha:
                 self.dnn.leaky_relu_alpha = leaky_alpha
                 self.result_text.append(f"Leaky ReLU alpha {leaky_alpha:.3f} olarak güncellendi.")


            return True

        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Hazırlık Hatası", f"DNN ve veri hazırlanırken hata: {str(e)}")
            return False

    def run_step(self):
        if self._prepare_dnn_and_data():
            try:
                loss, y_pred = self.dnn.train_step(self.current_x, self.current_y)
                self.total_steps += 1
                self.loss_history.append((self.total_steps, loss))
                self.result_text.append("-" * 20)
                self.result_text.append(f"[Adım {self.total_steps}] Giriş (X): {self.current_x.ravel()}")
                self.result_text.append(f"Hedef (Y): {self.current_y.ravel()}")
                self.result_text.append(f"Tahmin (Y_pred): {np.round(y_pred.ravel(), 4)}")
                output_act_info = self.dnn.activations[-1] if self.dnn and self.dnn.activations else "N/A"
                self.result_text.append(f"Kayıp ({self.dnn.loss_type} | Çıkış Akt: {output_act_info}): {loss:.6f}")
                self.result_text.append("")
                self.result_text.verticalScrollBar().setValue(self.result_text.verticalScrollBar().maximum())
            except Exception as e:
                traceback.print_exc()
                QMessageBox.critical(self, "Çalıştırma Hatası", f"Tek adım atılırken hata: {str(e)}")

    def run_multi_steps(self):
        if self._prepare_dnn_and_data():
            try:
                num_steps_to_run = self.steps_spinbox.value()
                initial_loss = -1; final_loss = -1; final_y_pred = None
                start_step = self.total_steps + 1
                self.result_text.append(f"--- {num_steps_to_run} Adımlık Eğitim Başlatılıyor (Adım {start_step}'den itibaren) ---")
                QApplication.processEvents()

                for i in range(num_steps_to_run):
                    current_step_number = self.total_steps + 1
                    loss, y_pred = self.dnn.train_step(self.current_x, self.current_y)
                    self.total_steps = current_step_number
                    self.loss_history.append((self.total_steps, loss))
                    if i == 0: initial_loss = loss
                    if i == num_steps_to_run - 1: final_loss = loss; final_y_pred = y_pred
                    log_interval = max(1, num_steps_to_run // 20) if num_steps_to_run > 20 else 1
                    if (i + 1) % log_interval == 0 and num_steps_to_run > 1:
                       self.result_text.append(f"  Adım {self.total_steps}/{start_step + num_steps_to_run - 1}, Kayıp: {loss:.6f}")
                       QApplication.processEvents()
                
                output_act_info = self.dnn.activations[-1] if self.dnn and self.dnn.activations else "N/A"
                self.result_text.append(f"--- {num_steps_to_run} Adımlık Eğitim Tamamlandı (Son Adım: {self.total_steps}) ---")
                self.result_text.append(f"Giriş (X): {self.current_x.ravel()}")
                self.result_text.append(f"Hedef (Y): {self.current_y.ravel()}")
                if final_y_pred is not None: self.result_text.append(f"Son Tahmin (Y_pred): {np.round(final_y_pred.ravel(), 4)}")
                self.result_text.append(f"Başlangıç Kaybı (Adım {start_step}): {initial_loss:.6f}")
                self.result_text.append(f"Son Kayıp (Adım {self.total_steps}): {final_loss:.6f}")
                self.result_text.append(f"Kayıp Fonksiyonu: {self.dnn.loss_type}, Çıkış Aktivasyonu: {output_act_info}")
                self.result_text.append("")
                self.result_text.verticalScrollBar().setValue(self.result_text.verticalScrollBar().maximum())

            except Exception as e:
                traceback.print_exc()
                QMessageBox.critical(self, "Çalıştırma Hatası", f"{num_steps_to_run} adım atılırken hata: {str(e)}")

    # --- Parametre Girişi ---
    def enter_parameters(self):
        try:
            # Mevcut yapı ve aktivasyonları UI'dan al
            num_inputs = self.input_spin.value()
            num_outputs = self.output_spin.value()
            
            hidden_nodes = [config['neurons'].value() for config in self.hidden_layer_configs]
            hidden_activations = [config['activation'].currentText() for config in self.hidden_layer_configs]
            output_activation = self.output_layer_activation_combo.currentText()
            
            layer_sizes = [num_inputs] + hidden_nodes + [num_outputs]
            current_activations = hidden_activations + [output_activation]

            if len(layer_sizes) < 2: QMessageBox.warning(self, "Hata", "Parametre girmek için geçerli yapı yok."); return

            dialog = QDialog(self); dialog.setWindowTitle("Parametre Girişi (Tek Tek)")
            dialog.setMinimumWidth(600); dialog.setMinimumHeight(400)
            scroll_area = QScrollArea(dialog); scroll_area.setWidgetResizable(True)
            scroll_content = QWidget(); params_layout = QVBoxLayout(scroll_content)
            self.manual_param_widgets = {}

            for l_idx_param in range(len(layer_sizes) - 1): # l_idx_param: 0'dan başlar, (W,b) setleri için
                layer_num_display = l_idx_param + 1 # Kullanıcıya gösterilen katman numarası (1'den başlar)
                prev_layer_size = layer_sizes[l_idx_param]
                current_layer_size = layer_sizes[l_idx_param+1]
                
                # Katman adı için aktivasyon bilgisini ekle
                act_func_for_layer = current_activations[l_idx_param]
                layer_group_title = f"Katman {layer_num_display} (L{l_idx_param} -> L{layer_num_display}, Akt: {act_func_for_layer})"
                layer_group = QGroupBox(layer_group_title)
                layer_grid = QGridLayout()

                w_key = f"W{layer_num_display}"; self.manual_param_widgets[w_key] = []
                layer_grid.addWidget(QLabel(f"<b>Ağırlıklar (W{layer_num_display})</b> [Hedef <- Kaynak]:"), 0, 0, 1, 4)
                row_offset = 1; col_width = 2
                for j in range(current_layer_size): # Hedef nöron
                    w_row_widgets = []
                    for k in range(prev_layer_size): # Kaynak nöron
                        label = QLabel(f"W[{j+1}<-{k+1}]:"); spinbox = QDoubleSpinBox()
                        spinbox.setRange(-100.0, 100.0); spinbox.setDecimals(4); spinbox.setSingleStep(0.01); spinbox.setFixedWidth(100)
                        default_val = np.random.randn() * np.sqrt(1. / prev_layer_size) # Basit bir başlatma
                        if self.dnn and w_key in self.dnn.parameters and \
                           j < self.dnn.parameters[w_key].shape[0] and \
                           k < self.dnn.parameters[w_key].shape[1]:
                             default_val = self.dnn.parameters[w_key][j, k]
                        spinbox.setValue(default_val)
                        grid_row, grid_col = row_offset + j, k * col_width
                        layer_grid.addWidget(label, grid_row, grid_col); layer_grid.addWidget(spinbox, grid_row, grid_col + 1)
                        w_row_widgets.append(spinbox)
                    self.manual_param_widgets[w_key].append(w_row_widgets)

                b_key = f"b{layer_num_display}"; self.manual_param_widgets[b_key] = []
                bias_row_start = row_offset + current_layer_size + 1
                layer_grid.addWidget(QLabel(f"<b>Biaslar (b{layer_num_display})</b> [Nöron]:"), bias_row_start -1 , 0, 1, 4)
                for j in range(current_layer_size):
                    label = QLabel(f"b[{j+1}]:"); spinbox = QDoubleSpinBox()
                    spinbox.setRange(-100.0, 100.0); spinbox.setDecimals(4); spinbox.setSingleStep(0.01); spinbox.setFixedWidth(100)
                    default_val = 0.0
                    if self.dnn and b_key in self.dnn.parameters and \
                       j < self.dnn.parameters[b_key].shape[0]:
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
                new_params = {}
                learning_rate = self.lr_spinbox.value() # UI'dan güncel LR
                loss_type = self.loss_combo.currentText() # UI'dan güncel kayıp
                leaky_alpha = self.leaky_relu_alpha_spinbox.value() # UI'dan güncel alpha

                try:
                    for l_param_idx in range(len(layer_sizes) - 1):
                        layer_num_display = l_param_idx + 1
                        w_key = f"W{layer_num_display}"; b_key = f"b{layer_num_display}"
                        rows, cols = layer_sizes[l_param_idx+1], layer_sizes[l_param_idx] # Hedef, Kaynak
                        W = np.zeros((rows, cols)); b = np.zeros((rows, 1))
                        for j in range(rows):
                            for k in range(cols): W[j, k] = self.manual_param_widgets[w_key][j][k].value()
                        for j in range(rows): b[j, 0] = self.manual_param_widgets[b_key][j].value()
                        new_params[w_key] = W; new_params[b_key] = b
                    
                    # DNN'i güncel konfigürasyon ve parametrelerle oluştur/güncelle
                    self.dnn = DeepDNN(layer_sizes,
                                       learning_rate=learning_rate,
                                       activations=current_activations,
                                       loss_type=loss_type,
                                       leaky_relu_alpha=leaky_alpha)
                    self.dnn.parameters = new_params # Manuel girilen parametreleri ata
                    self.network_config = layer_sizes # network_config'i de güncelle
                    self.activations_config = current_activations # activations_config'i de güncelle
                    self.loss_history = []
                    self.total_steps = 0
                    QMessageBox.information(self, "Başarılı", "Parametreler ağa yüklendi. Ağ ve eğitim geçmişi sıfırlandı.")
                    self.result_text.append(f"Manuel parametreler ağa yüklendi. Adım sayacı ve kayıp geçmişi sıfırlandı.")
                    self.manual_param_widgets = {}
                except Exception as parse_error:
                     traceback.print_exc(); QMessageBox.critical(self, "Parametre Yükleme Hatası", f"Hata: {parse_error}")
            else: self.manual_param_widgets = {}
        except Exception as e:
            traceback.print_exc(); QMessageBox.warning(self, "Parametre Girişi Hatası", f"Diyalog hatası: {str(e)}")

    # --- Görselleştirme ve Sıfırlama ---
    def visualize_architecture(self):
        try:
            params = None; layer_sizes = []
            current_activations = []

            if self.dnn is None:
                # DNN yoksa, UI'dan mevcut konfigürasyonu al
                num_inputs = self.input_spin.value()
                num_outputs = self.output_spin.value()
                hidden_nodes = [config['neurons'].value() for config in self.hidden_layer_configs]
                hidden_acts = [config['activation'].currentText() for config in self.hidden_layer_configs]
                output_act = self.output_layer_activation_combo.currentText()
                
                layer_sizes = [num_inputs] + hidden_nodes + [num_outputs]
                current_activations = hidden_acts + [output_act]
                params = None # Parametreler henüz yok
                QMessageBox.information(self, "Bilgi","Ağırlık/Bias görmek için ağı başlatın veya parametre girin.");
            else:
                layer_sizes = self.dnn.layer_sizes
                params = self.dnn.parameters
                current_activations = self.dnn.activations # DNN'den al

            if not layer_sizes or len(layer_sizes) < 2: QMessageBox.warning(self, "Hata", "Görselleştirilecek yapı yok."); return
            if len(current_activations) != len(layer_sizes) -1:
                 QMessageBox.warning(self, "Hata", "Katman sayısı ile aktivasyon sayısı uyuşmuyor. Lütfen ağı sıfırlayıp tekrar deneyin.")
                 return


            max_neurons = max(layer_sizes) if layer_sizes else 1; num_layers_viz = len(layer_sizes)
            fig_width = max(12, num_layers_viz * 3.5); fig_height = max(8, max_neurons * 1.2)
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            neuron_positions = []; node_radius = 0.4; h_spacing = 4.0; v_spacing = 1.2

            for i, layer_size_val in enumerate(layer_sizes): # i: katman indeksi (0=giriş, 1=ilk gizli/çıkış, ...)
                total_height = (layer_size_val - 1) * v_spacing
                y_start = (fig_height - total_height) / 2.0 - node_radius * max_neurons *0.1
                if layer_size_val == 1 : y_start = fig_height / 2.0

                layer_positions_current = []
                layer_name_prefix = ""
                act_func_name = ""

                if i == 0: # Giriş katmanı
                    layer_name_prefix = "Giriş"
                else: # Gizli veya çıkış katmanları (aktivasyonları var)
                    act_func_name = f" ({current_activations[i-1]})" # i-1 çünkü activations[0] ilk W,b setine (L1'e) ait
                    if i == num_layers_viz - 1: layer_name_prefix = "Çıkış"
                    else: layer_name_prefix = f"Gizli {i}"
                
                ax.text(i * h_spacing, y_start + total_height + v_spacing * 1.5, f"{layer_name_prefix} (L{i})\n{layer_size_val} nöron{act_func_name}", ha='center', va='bottom', fontsize=9, weight='bold')

                for j in range(layer_size_val):
                    x = i * h_spacing; y = y_start + j * v_spacing
                    circle = plt.Circle((x, y), node_radius, color='skyblue', zorder=4, ec='black')
                    ax.add_patch(circle); ax.text(x, y, f"N{j+1}", ha='center', va='center', zorder=5, fontsize=7)
                    layer_positions_current.append({'pos': (x, y), 'id': j})
                    if params and i > 0: # i > 0 ise bu bir gizli veya çıkış katmanıdır, biasları vardır
                        bias_key = f"b{i}" # W1/b1, W2/b2...
                        if bias_key in params and params[bias_key].shape[0] > j:
                            bias_val = params[bias_key][j, 0]; ax.text(x, y - node_radius * 1.5, f"b={bias_val:.2f}",ha='center', va='top', zorder=5, fontsize=6, color='red')
                neuron_positions.append(layer_positions_current)

            if params: # Ağırlıklar
                for l_idx_param in range(num_layers_viz - 1): # l_idx_param: kaynak katman (0'dan başlar)
                    weight_key = f"W{l_idx_param+1}" # Ağırlık matrisi W1, W2, ...
                    if weight_key in params:
                        W = params[weight_key] # W[hedef_nöron_idx, kaynak_nöron_idx]
                        for src_neuron_idx_in_layer, src_neuron_data in enumerate(neuron_positions[l_idx_param]):
                            for dst_neuron_idx_in_layer, dst_neuron_data in enumerate(neuron_positions[l_idx_param + 1]):
                                if dst_neuron_idx_in_layer < W.shape[0] and src_neuron_idx_in_layer < W.shape[1]:
                                    src_pos, dst_pos = src_neuron_data['pos'], dst_neuron_data['pos']
                                    weight_val = W[dst_neuron_idx_in_layer, src_neuron_idx_in_layer]
                                    arrow = FancyArrowPatch(src_pos, dst_pos, arrowstyle='->', mutation_scale=10, color='gray', lw=0.5, zorder=1, shrinkA=node_radius*1.2, shrinkB=node_radius*1.2)
                                    ax.add_patch(arrow)
                                    mid_x = (src_pos[0] + dst_pos[0]) / 2
                                    mid_y = (src_pos[1] + dst_pos[1]) / 2
                                    offset_x = (dst_pos[0] - src_pos[0]) * 0.1 # Kaydırma için
                                    offset_y = (dst_pos[1] - src_pos[1]) * 0.1 # Kaydırma için
                                    ax.text(mid_x - offset_x, mid_y - offset_y, f"w={weight_val:.2f}", fontsize=6, color='blue', ha='center', va='center', zorder=3, bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.7, ec='none'))
            else: # Sadece bağlantı okları (parametreler yoksa)
                 for l_idx_param in range(num_layers_viz - 1):
                     for src_neuron_data in neuron_positions[l_idx_param]:
                         for dst_neuron_data in neuron_positions[l_idx_param + 1]:
                             arrow = FancyArrowPatch(src_neuron_data['pos'], dst_neuron_data['pos'], arrowstyle='->', mutation_scale=10, color='lightgray', lw=0.5, zorder=1, shrinkA=node_radius*1.2, shrinkB=node_radius*1.2)
                             ax.add_patch(arrow)

            ax.set_xlim(-node_radius*2, (num_layers_viz - 1) * h_spacing + node_radius*2)
            all_y_coords = [n['pos'][1] for layer in neuron_positions for n in layer] if neuron_positions and any(neuron_positions) else [0]
            min_y, max_y = min(all_y_coords) if all_y_coords else 0, max(all_y_coords) if all_y_coords else 1
            ax.set_ylim(min_y - v_spacing * 2 , max_y + v_spacing * 2.5)
            ax.axis('off')
            plt.title("DNN Mimarisi - Ağırlıklar ve Biaslar (varsa)", fontsize=12)
            plt.tight_layout(pad=1.0);
            plt.show(block=False)
        except Exception as e:
            traceback.print_exc(); QMessageBox.warning(self, "Görselleştirme Hatası", f"Hata: {str(e)}")


    def plot_loss_graph(self):
        if not self.loss_history:
            QMessageBox.information(self, "Bilgi", "Gösterilecek kayıp geçmişi bulunmuyor. Lütfen önce ağı eğitin.")
            return
        try:
            steps, losses = zip(*self.loss_history)
            plt.figure(figsize=(10, 6))
            plt.plot(steps, losses, marker='.', linestyle='-', markersize=4)
            plt.xlabel("Eğitim Adımı Sayısı")
            plt.ylabel("Kayıp Değeri")
            loss_title = self.dnn.loss_type if self.dnn else "Bilinmiyor"
            output_act_title = self.dnn.activations[-1] if self.dnn and self.dnn.activations else "Bilinmiyor"
            plt.title(f"Eğitim Kayıp Grafiği (Kayıp: {loss_title}, Çıkış Akt: {output_act_title})")
            plt.grid(True)
            plt.tight_layout()
            plt.show(block=False)
        except Exception as e:
             traceback.print_exc(); QMessageBox.warning(self, "Grafik Hatası", f"Kayıp grafiği çizilirken hata: {str(e)}")

    def reset_network(self, show_message=True):
        self.dnn = None
        self.network_config = None
        self.activations_config = None 
        self.loss_history = []
        self.total_steps = 0
        self.current_x = None
        self.current_y = None
        self.result_text.clear()
        self.result_text.append("Ağ durumu, sonuçlar ve kayıp geçmişi sıfırlandı.")
        if show_message:
             QMessageBox.information(self, "Sıfırlandı", "Yapay sinir ağı durumu, sonuçlar ve kayıp geçmişi sıfırlandı.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DNNConfigurator()
    window.show()
    sys.exit(app.exec_())