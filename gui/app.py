
import sys
from PyQt5.QtWidgets import *
from main import query_system

class CodeBrainGUI(QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("CodeBrain BMS Analyzer")

        layout = QVBoxLayout()

        self.input = QTextEdit()
        self.input.setPlaceholderText("Enter BMS diagnostic question")

        self.button = QPushButton("Analyze")

        self.output = QTextEdit()

        self.button.clicked.connect(self.run_analysis)

        layout.addWidget(self.input)
        layout.addWidget(self.button)
        layout.addWidget(self.output)

        self.setLayout(layout)

    def run_analysis(self):

        question = self.input.toPlainText()

        result = query_system(question)

        self.output.setText(result)


app = QApplication(sys.argv)

window = CodeBrainGUI()

window.show()

sys.exit(app.exec_())
