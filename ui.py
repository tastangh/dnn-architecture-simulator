from PyQt5.QtWidgets import (
    QWidget, QLabel, QSpinBox, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QGridLayout, QComboBox, QTextEdit,
    QFormLayout, QMessageBox
)
from backend import SimpleDNN
import numpy as np

class DNNConfigurator(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DNN Architecture Simulator")
        self.hidden_layer_inputs = []
        self.input_boxes = []
        self.network_config = None
        self.dnn = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.addWidget(QLabel("<h2>Ağ Yapılandırması</h2>"))

        # Input sayısı
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Input sayısı:"))
        self.input_spin = QSpinBox()
        self.input_spin.setValue(3)
        self.input_spin.valueChanged.connect(self.update_input_vector_fields)
        input_layout.addWidget(self.input_spin)
        layout.addLayout(input_layout)

        # Output sayısı
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output sayısı:"))
        self.output_spin = QSpinBox()
        self.output_spin.setValue(1)
        output_layout.addWidget(self.output_spin)
        layout.addLayout(output_layout)

        # Hidden layer sayısı
        hidden_layout = QHBoxLayout()
        hidden_layout.addWidget(QLabel("Hidden Layer sayısı:"))
        self.hidden_spin = QSpinBox()
        self.hidden_spin.setValue(2)
        self.hidden_spin.valueChanged.connect(self.update_hidden_layer_inputs)
        hidden_layout.addWidget(self.hidden_spin)
        layout.addLayout(hidden_layout)

        # Hidden layer girişleri
        self.hidden_grid = QGridLayout()
        layout.addLayout(self.hidden_grid)
        self.update_hidden_layer_inputs()

        # Dinamik input vektör girişi
        layout.addWidget(QLabel("Input Vektörü:"))
        self.input_dynamic_layout = QVBoxLayout()
        layout.addLayout(self.input_dynamic_layout)
        self.update_input_vector_fields()

        # Aktivasyon ve loss (şimdilik sabit)
        self.act_combo = QComboBox()
        self.act_combo.addItems(["Sigmoid"])
        self.loss_combo = QComboBox()
        self.loss_combo.addItems(["MSE"])
        layout.addWidget(QLabel("Aktivasyon Fonksiyonu:"))
        layout.addWidget(self.act_combo)
        layout.addWidget(QLabel("Loss Fonksiyonu:"))
        layout.addWidget(self.loss_combo)

        # Target vektörü
        self.target_line = QLineEdit("1.0")
        form_layout = QFormLayout()
        form_layout.addRow("Target Vektörü:", self.target_line)
        layout.addLayout(form_layout)

        # Adım at düğmesi
        self.step_button = QPushButton("Adım At")
        self.step_button.clicked.connect(self.run_step)
        layout.addWidget(self.step_button)

        # Çıktı alanı
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        layout.addWidget(self.result_text)

        self.setLayout(layout)

    def update_hidden_layer_inputs(self):
        while self.hidden_grid.count():
            item = self.hidden_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.hidden_layer_inputs = []
        for i in range(self.hidden_spin.value()):
            label = QLabel(f"Hidden {i+1}:")
            spin = QSpinBox()
            spin.setValue(4)
            self.hidden_grid.addWidget(label, i, 0)
            self.hidden_grid.addWidget(spin, i, 1)
            self.hidden_layer_inputs.append(spin)

    def update_input_vector_fields(self):
        # Önce tüm içeriği temizle
        while self.input_dynamic_layout.count():
            child = self.input_dynamic_layout.takeAt(0)
            if child.layout():
                sub_layout = child.layout()
                while sub_layout.count():
                    item = sub_layout.takeAt(0)
                    widget = item.widget()
                    if widget is not None:
                        widget.deleteLater()
            elif child.widget():
                child.widget().deleteLater()

        self.input_boxes = []

        input_count = self.input_spin.value()
        for i in range(input_count):
            h_layout = QHBoxLayout()
            label = QLabel(f"x{i+1}:")
            line_edit = QLineEdit("0.0")
            h_layout.addWidget(label)
            h_layout.addWidget(line_edit)
            self.input_dynamic_layout.addLayout(h_layout)
            self.input_boxes.append(line_edit)


    def run_step(self):
        try:
            num_inputs = self.input_spin.value()
            num_outputs = self.output_spin.value()
            hidden = [spin.value() for spin in self.hidden_layer_inputs]
            self.network_config = [num_inputs] + hidden + [num_outputs]
            if self.dnn is None:
                self.dnn = SimpleDNN(self.network_config)

            x_vals = []
            for combo, line in self.input_boxes:
                try:
                    val = float(line.text())
                    x_vals.append(val)
                except ValueError:
                    raise ValueError(f"Geçersiz sayı girdiniz: '{line.text()}'")
            x = np.array(x_vals).reshape(-1, 1)

            y = np.array([float(s.strip()) for s in self.target_line.text().split(",")]).reshape(-1, 1)

            loss, y_pred = self.dnn.train_step(x, y)
            self.result_text.append("-----")
            self.result_text.append(f"Input: {x.ravel()}")
            self.result_text.append(f"Output: {y_pred.ravel()}")
            self.result_text.append(f"Target: {y.ravel()}")
            self.result_text.append(f"Loss: {loss:.6f}\n")
        except Exception as e:
            QMessageBox.warning(self, "Hata", str(e))
