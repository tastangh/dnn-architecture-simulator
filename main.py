from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QSpinBox, QVBoxLayout,
    QHBoxLayout, QPushButton, QLineEdit, QGridLayout, QScrollArea
)
import sys

class DNNConfigurator(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DNN Architecture Simulator")
        self.hidden_layer_inputs = []
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Input sayısı
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Input sayısı:"))
        self.input_spin = QSpinBox()
        self.input_spin.setMinimum(1)
        input_layout.addWidget(self.input_spin)
        layout.addLayout(input_layout)

        # Output sayısı
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output sayısı:"))
        self.output_spin = QSpinBox()
        self.output_spin.setMinimum(1)
        output_layout.addWidget(self.output_spin)
        layout.addLayout(output_layout)

        # Hidden layer sayısı
        hidden_layout = QHBoxLayout()
        hidden_layout.addWidget(QLabel("Hidden Layer sayısı:"))
        self.hidden_spin = QSpinBox()
        self.hidden_spin.setMinimum(1)
        self.hidden_spin.valueChanged.connect(self.update_hidden_layer_inputs)
        hidden_layout.addWidget(self.hidden_spin)
        layout.addLayout(hidden_layout)

        # Hidden layer giriş alanları
        self.hidden_grid = QGridLayout()
        layout.addLayout(self.hidden_grid)

        # Devam butonu
        self.next_button = QPushButton("Devam Et")
        self.next_button.clicked.connect(self.collect_parameters)
        layout.addWidget(self.next_button)

        self.setLayout(layout)

    def update_hidden_layer_inputs(self):
        # Temizle
        for i in reversed(range(self.hidden_grid.count())):
            self.hidden_grid.itemAt(i).widget().deleteLater()
        self.hidden_layer_inputs = []

        count = self.hidden_spin.value()
        for i in range(count):
            label = QLabel(f"Hidden Layer {i+1} nöron sayısı:")
            spin = QSpinBox()
            spin.setMinimum(1)
            self.hidden_grid.addWidget(label, i, 0)
            self.hidden_grid.addWidget(spin, i, 1)
            self.hidden_layer_inputs.append(spin)

    def collect_parameters(self):
        num_inputs = self.input_spin.value()
        num_outputs = self.output_spin.value()
        hidden_sizes = [spin.value() for spin in self.hidden_layer_inputs]

        print(f"Giriş sayısı: {num_inputs}")
        print(f"Çıkış sayısı: {num_outputs}")
        print(f"Hidden katmanlar: {hidden_sizes}")
        # Buradan dnn_config.py'ye yazılabilir veya bir sonraki ekrana geçilebilir

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DNNConfigurator()
    window.show()
    sys.exit(app.exec_())
