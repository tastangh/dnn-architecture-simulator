from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QSpinBox, QLineEdit, QPushButton, QComboBox, QFrame
)
from PyQt5.QtCore import Qt
import sys

class DNNSimulator(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DNN Architecture Simulator")
        self.setFixedSize(960, 540)  # Sabit pencere boyutu
        self.input_fields = []

        self.build_ui()

    def build_ui(self):
        main_layout = QVBoxLayout()

        # --- Input/Output Ayarları ---
        input_section = QVBoxLayout()
        input_section.addWidget(QLabel("<b>Input/Output Ayarları</b>"))

        h_input = QHBoxLayout()
        h_input.addWidget(QLabel("Input sayısı:"))
        self.input_count_spin = QSpinBox()
        self.input_count_spin.setMinimum(1)
        self.input_count_spin.setMaximum(10)
        self.input_count_spin.valueChanged.connect(self.update_input_fields)
        h_input.addWidget(self.input_count_spin)
        input_section.addLayout(h_input)

        self.inputs_layout = QVBoxLayout()
        input_section.addLayout(self.inputs_layout)

        main_layout.addLayout(input_section)
        main_layout.addWidget(self._separator())

        # --- Output sayısı ---
        h_output = QHBoxLayout()
        h_output.addWidget(QLabel("Output sayısı:"))
        self.output_count_spin = QSpinBox()
        self.output_count_spin.setMinimum(1)
        self.output_count_spin.setMaximum(10)
        h_output.addWidget(self.output_count_spin)
        main_layout.addLayout(h_output)

        main_layout.addWidget(self._separator())

        # --- Gizli Katman Ayarları ---
        hidden_section = QVBoxLayout()
        hidden_section.addWidget(QLabel("<b>Gizli Katman Ayarları</b>"))

        self.hidden_layer_spin = QSpinBox()
        self.hidden_layer_spin.setMinimum(1)
        self.hidden_layer_spin.setMaximum(5)
        hidden_section.addWidget(QLabel("Hidden Layer sayısı:"))
        hidden_section.addWidget(self.hidden_layer_spin)

        self.hidden_neuron_count = []
        for i in range(5):
            neuron_spin = QSpinBox()
            neuron_spin.setMinimum(1)
            neuron_spin.setValue(4)
            neuron_spin.setPrefix(f"Katman {i+1} nöron sayısı: ")
            self.hidden_neuron_count.append(neuron_spin)
            hidden_section.addWidget(neuron_spin)

        main_layout.addLayout(hidden_section)
        main_layout.addWidget(self._separator())

        # --- Aktivasyon ve Loss Seçimi ---
        activation_loss_layout = QVBoxLayout()
        activation_loss_layout.addWidget(QLabel("<b>Aktivasyon ve Loss Fonksiyonu</b>"))

        self.activation_combo = QComboBox()
        self.activation_combo.addItems(["Sigmoid", "ReLU", "Tanh", "Linear"])
        activation_loss_layout.addWidget(QLabel("Aktivasyon Fonksiyonu:"))
        activation_loss_layout.addWidget(self.activation_combo)

        self.loss_combo = QComboBox()
        self.loss_combo.addItems(["MSE", "MAE", "Binary Crossentropy"])
        activation_loss_layout.addWidget(QLabel("Loss Fonksiyonu:"))
        activation_loss_layout.addWidget(self.loss_combo)

        main_layout.addLayout(activation_loss_layout)
        main_layout.addWidget(self._separator())

        # --- Simülasyon Butonu ---
        self.simulate_button = QPushButton("Simülasyonu Başlat")
        self.simulate_button.clicked.connect(self.run_simulation)
        main_layout.addWidget(self.simulate_button, alignment=Qt.AlignCenter)

        self.setLayout(main_layout)

    def _separator(self):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        return line

    def update_input_fields(self):
        for field in self.input_fields:
            self.inputs_layout.removeWidget(field)
            field.deleteLater()
        self.input_fields.clear()

        count = self.input_count_spin.value()
        for i in range(count):
            inp = QLineEdit()
            inp.setPlaceholderText(f"Input {i+1} değeri")
            self.inputs_layout.addWidget(inp)
            self.input_fields.append(inp)

    def run_simulation(self):
        inputs = [f.text() for f in self.input_fields]
        print("Girilen input değerleri:", inputs)
        print("Aktivasyon:", self.activation_combo.currentText())
        print("Loss:", self.loss_combo.currentText())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DNNSimulator()
    window.show()
    sys.exit(app.exec_())
