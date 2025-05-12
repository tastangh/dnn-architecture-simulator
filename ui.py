from PyQt5.QtWidgets import (
    QWidget, QLabel, QSpinBox, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QGridLayout, QTextEdit, QMessageBox, QGroupBox,
    QScrollArea, QDialog, QFormLayout, QDialogButtonBox, QComboBox
)

from backend import DeepDNN
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch

class DNNConfigurator(QWidget):
    def __init__(self):
        self.activation_combo = QComboBox()
        self.activation_combo.addItems(["relu", "sigmoid"])
        self.loss_combo = QComboBox()
        self.loss_combo.addItems(["mse", "mae", "rmse", "cross_entropy"])

        super().__init__()
        self.setWindowTitle("DNN Architecture Simulator")
        self.resize(1200, 800)

        self.input_spin = QSpinBox()
        self.output_spin = QSpinBox()
        self.hidden_spin = QSpinBox()

        self.input_boxes = []
        self.hidden_layer_inputs = []
        self.output_boxes = []
        self.manual_weights = []
        self.manual_biases = []

        self.result_text = QTextEdit()
        self.network_config = None
        self.dnn = None

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # INPUT CONFIGURATION
        input_group = QGroupBox("Input Yapılandırması")
        input_layout = QVBoxLayout()
        input_count_layout = QHBoxLayout()
        input_count_layout.addWidget(QLabel("Input Sayısı:"))
        self.input_spin.setMinimum(1)
        self.input_spin.setValue(5)
        self.input_spin.valueChanged.connect(self.update_input_boxes)
        input_count_layout.addWidget(self.input_spin)
        input_layout.addLayout(input_count_layout)
        self.input_grid = QGridLayout()
        input_layout.addLayout(self.input_grid)
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        self.update_input_boxes()

        # OUTPUT CONFIGURATION
        output_group = QGroupBox("Output Yapılandırması")
        output_layout = QVBoxLayout()
        output_count_layout = QHBoxLayout()
        output_count_layout.addWidget(QLabel("Output Sayısı:"))
        self.output_spin.setMinimum(1)
        self.output_spin.setValue(2)
        self.output_spin.valueChanged.connect(self.update_output_boxes)
        output_count_layout.addWidget(self.output_spin)
        output_layout.addLayout(output_count_layout)
        self.output_grid = QGridLayout()
        output_layout.addLayout(self.output_grid)
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        self.update_output_boxes()

        # HIDDEN CONFIGURATION
        hidden_group = QGroupBox("Hidden Layer Yapılandırması")
        hidden_layout = QVBoxLayout()
        hidden_count_layout = QHBoxLayout()
        hidden_count_layout.addWidget(QLabel("Hidden Layer Sayısı:"))
        self.hidden_spin.setMinimum(1)
        self.hidden_spin.setValue(2)
        self.hidden_spin.valueChanged.connect(self.update_hidden_inputs)
        hidden_count_layout.addWidget(self.hidden_spin)
        hidden_layout.addLayout(hidden_count_layout)
        self.hidden_grid = QGridLayout()
        hidden_layout.addLayout(self.hidden_grid)
        hidden_group.setLayout(hidden_layout)
        layout.addWidget(hidden_group)
        self.update_hidden_inputs()

        # WEIGHT/BIAS BUTTON
        param_btn = QPushButton("Ağırlık/Bias Girişi")
        param_btn.clicked.connect(self.enter_parameters)
        layout.addWidget(param_btn)
        # Aktivasyon ve Kayıp Fonksiyonu Seçimi
        act_loss_group = QGroupBox("Aktivasyon ve Kayıp Fonksiyonu")
        act_loss_layout = QHBoxLayout()
        act_loss_layout.addWidget(QLabel("Hidden Activation:"))
        act_loss_layout.addWidget(self.activation_combo)
        act_loss_layout.addWidget(QLabel("Loss Function:"))
        act_loss_layout.addWidget(self.loss_combo)
        act_loss_group.setLayout(act_loss_layout)
        layout.addWidget(act_loss_group)
        # RUN BUTTON
        btn = QPushButton("Adım At")
        btn.clicked.connect(self.run_step)
        layout.addWidget(btn)

        # VISUALIZATION BUTTON
        diagram_btn = QPushButton("Ağ Yapısını Görselleştir")
        diagram_btn.clicked.connect(self.visualize_architecture)
        layout.addWidget(diagram_btn)

        # RESULTS
        self.result_text.setReadOnly(True)
        layout.addWidget(self.result_text)

        scroll = QScrollArea()
        container = QWidget()
        container.setLayout(layout)
        scroll.setWidget(container)
        scroll.setWidgetResizable(True)

        main_layout = QVBoxLayout()
        main_layout.addWidget(scroll)
        self.setLayout(main_layout)

    def update_input_boxes(self):
        while self.input_grid.count():
            item = self.input_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.input_boxes = []
        cols = 4
        for i in range(self.input_spin.value()):
            label = QLabel(f"x{i+1}:")
            edit = QLineEdit("0.0")
            row = i // cols
            col = (i % cols) * 2
            self.input_grid.addWidget(label, row, col)
            self.input_grid.addWidget(edit, row, col + 1)
            self.input_boxes.append(edit)

    def update_output_boxes(self):
        while self.output_grid.count():
            item = self.output_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.output_boxes = []
        cols = 4
        for i in range(self.output_spin.value()):
            label = QLabel(f"y{i+1}:")
            edit = QLineEdit("1.0")
            row = i // cols
            col = (i % cols) * 2
            self.output_grid.addWidget(label, row, col)
            self.output_grid.addWidget(edit, row, col + 1)
            self.output_boxes.append(edit)

    def update_hidden_inputs(self):
        while self.hidden_grid.count():
            item = self.hidden_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.hidden_layer_inputs = []
        cols = 4
        for i in range(self.hidden_spin.value()):
            label = QLabel(f"Layer {i+1} Nöron Sayısı:")
            spin = QSpinBox()
            spin.setMinimum(1)
            spin.setValue(4)
            row = i // cols
            col = (i % cols) * 2
            self.hidden_grid.addWidget(label, row, col)
            self.hidden_grid.addWidget(spin, row, col + 1)
            self.hidden_layer_inputs.append(spin)

    def visualize_architecture(self):
        try:
            layer_sizes = [self.input_spin.value()] + [spin.value() for spin in self.hidden_layer_inputs] + [self.output_spin.value()]
            max_neurons = max(layer_sizes)
            fig, ax = plt.subplots(figsize=(12, 6))

            neuron_positions = []

            for i, layer_size in enumerate(layer_sizes):
                y_offset = (max_neurons - layer_size) / 2
                layer_positions = []
                for j in range(layer_size):
                    x = i * 2
                    y = max_neurons - j - y_offset
                    circle = plt.Circle((x, y), 0.3, color='skyblue', zorder=2)
                    ax.add_patch(circle)
                    ax.text(x, y, f"{j+1}", ha='center', va='center', zorder=3, fontsize=8)
                    layer_positions.append((x, y))
                neuron_positions.append(layer_positions)
                ax.text(x, max_neurons + 1, f"Layer {i+1}\n({layer_size} neurons)", ha='center', fontsize=9)

            for l in range(len(neuron_positions) - 1):
                for src in neuron_positions[l]:
                    for dst in neuron_positions[l + 1]:
                        arrow = FancyArrowPatch(src, dst, arrowstyle='->', mutation_scale=10, color='gray', lw=0.5, zorder=1)
                        ax.add_patch(arrow)

            ax.set_xlim(-1, len(layer_sizes) * 2)
            ax.set_ylim(-1, max_neurons + 3)
            ax.set_aspect('equal')
            ax.axis('off')
            plt.title("DNN Katman Yapısı (nöronlar + oklar)")
            plt.tight_layout()
            plt.show()

        except Exception as e:
            QMessageBox.warning(self, "Görselleştirme Hatası", str(e))

    def enter_parameters(self):
        try:
            dialog = QDialog(self)
            dialog.setWindowTitle("Ağırlık ve Bias Girişi")
            form_layout = QFormLayout(dialog)

            layer_sizes = [self.input_spin.value()] + [spin.value() for spin in self.hidden_layer_inputs] + [self.output_spin.value()]
            self.manual_weights = []
            self.manual_biases = []

            for l in range(len(layer_sizes) - 1):
                w = QLineEdit("0.1")
                b = QLineEdit("0.0")
                form_layout.addRow(f"Layer {l+1} Weight Matrix ({layer_sizes[l+1]}x{layer_sizes[l]}):", w)
                form_layout.addRow(f"Layer {l+1} Bias Vector ({layer_sizes[l+1]}):", b)
                self.manual_weights.append(w)
                self.manual_biases.append(b)

            buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            buttons.accepted.connect(dialog.accept)
            buttons.rejected.connect(dialog.reject)
            form_layout.addWidget(buttons)

            if dialog.exec_():
                QMessageBox.information(self, "Başarılı", "Ağırlık ve bias değerleri kaydedildi (şimdilik sadece görüntüleme amaçlı).")

        except Exception as e:
            QMessageBox.warning(self, "Parametre Girişi Hatası", str(e))

    def run_step(self):
        try:
            num_inputs = self.input_spin.value()
            num_outputs = self.output_spin.value()
            hidden = [spin.value() for spin in self.hidden_layer_inputs]
            self.network_config = [num_inputs] + hidden + [num_outputs]

            x_vals = [float(box.text()) for box in self.input_boxes]
            x = np.array(x_vals).reshape(-1, 1)

            y_vals = [float(box.text()) for box in self.output_boxes]
            y = np.array(y_vals).reshape(-1, 1)

            activation = self.activation_combo.currentText()
            loss_type = self.loss_combo.currentText()

            if self.dnn is None:
                self.dnn = DeepDNN(self.network_config, learning_rate=0.01,
                                hidden_activation=activation,
                                loss_type=loss_type)

            loss, y_pred = self.dnn.train_step(x, y)

            self.result_text.append("-----")
            self.result_text.append(f"Input: {x.ravel()}")
            self.result_text.append(f"Output: {y_pred.ravel()}")
            self.result_text.append(f"Loss ({loss_type}): {loss:.6f}\n")

        except Exception as e:
            QMessageBox.warning(self, "Hata", str(e))
