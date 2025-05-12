import sys
from PyQt5.QtWidgets import QApplication
from ui import DNNConfigurator

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DNNConfigurator()
    window.show()
    sys.exit(app.exec_())