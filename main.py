import sys
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
from PIL import Image, ImageDraw


class Perceptron:
    def __init__(self, activation_function="step"):
        self.activation_function = activation_function
        self.weights = np.random.rand(3) - 0.5
        self.x = []
        self.y = []
        self.learning_rate = 0.05
        self.current_index = 0
        self.generations = 1
        self.current_generation = 1
        self.total_points = 0
        self.current_total_error = 0.0
        self.stop = False
        self.img_size = (400, 400)
        self.img = None

    def randomize_w(self):
        self.weights = np.random.rand(3) - 0.5

    def clear(self):
        self.x = []
        self.y = []
        self.current_index = 0
        self.current_total_error = 0
        self.current_generation = 1
        self.total_points = 0

    def set_activation_step(self):
        self.activation_function = "step"

    def set_activation_sigmoid(self):
        self.activation_function = "sigmoid"

    def check_sizes(self):
        return True if len(self.x) == len(self.y) else False

    def activate(self, x):
        if (self.activation_function == "step"):
            return 1 if x >= 0 else 0
        if (self.activation_function == "sigmoid"):
            return self.sigmoid(x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def error_function(self):
        total_error = 0
        if (self.activation_function == "step"):
            for i in range(len(self.x)):
                _, fwd = self.forward_pass(self.x[i][0], self.x[i][1])
                if self.y[i] != fwd:
                    total_error += 1
        if (self.activation_function == "sigmoid"):
            for i in range(len(self.x)):
                _, fwd = self.forward_pass(self.x[i][0], self.x[i][1])
                total_error += ((self.y[i] - fwd)**2)/2
                #total_error += self.log_loss(self.x[i][0], self.x[i][1], self.y[i], fwd)
            #total_error = total_error * (1 / len(self.x))
        return total_error

    def log_loss(self, x1, x2, y, y_pred):
        return -(1 - y) * np.log(1 - y_pred) - y * np.log(y_pred)

    def gradient(self, x, y, y_pred, mult):
        d_log_loss = -(1 - y) / np.log(1 - y_pred) - y / np.log(y_pred)
        d_mse = y_pred - y
        d_activation = self.sigmoid_derivative(mult)
        d_weight = x
        return d_mse * d_activation * d_weight

    def forward_pass(self, x1, x2):
        i = [float(x1), float(x2), 1.0]
        i = np.array(i)
        mult = np.dot(self.weights, i)
        out = self.activate(mult)
        return mult, out

    def train(self, x, y, y_pred, mult):
        if (self.activation_function == "step"):
            if (y != y_pred):
                signal = 1 if y_pred == 0 else -1
                for i in range(2):
                    self.weights[i] = self.weights[i] + signal * self.learning_rate * x[i]
                self.weights[2] = self.weights[2] + signal * self.learning_rate
        if (self.activation_function == "sigmoid"):
            for i in range(2):
                self.weights[i] = self.weights[i] - self.learning_rate * self.gradient(x[i], y, y_pred, mult)
            self.weights[2] = self.weights[2] - self.learning_rate * self.gradient(1, y, y_pred, mult)

    def add_point(self, x_screen, y_screen, truth):
        x = x_screen / self.img_size[0]
        y = y_screen / self.img_size[1]
        if (type(x) is float and type(y) is float and type(truth) is int and (truth == 0 or truth == 1)):
            if (x >= 0 and x <= 1 and y >= 0 and y <= 1):
                self.x.append([x, y])
                self.y.append(truth)
        self.total_points = len(self.x)

    def generate_img(self, resolution = 10):
        x_dim = self.img_size[0]
        y_dim = self.img_size[1]
        img = np.ndarray((x_dim, y_dim, 3), dtype=np.uint8)
        x = 0
        while(x < x_dim):
            y = 0
            while(y < y_dim):
                _, y_pred = self.forward_pass(x / x_dim, (y / y_dim))
                for i in range(resolution):
                    for j in range(resolution):
                        img[y_dim -y -1 -i][x + j][0] = int((1 - y_pred)*255)
                        img[y_dim -y -1 -i][x + j][1] = 25
                        img[y_dim -y -1 -i][x + j][2] = int(y_pred*255)
                y += resolution
            else:
                x += resolution
        img = Image.fromarray(img)
        #img.format = "PNG"
        draw = ImageDraw.Draw(img)
        for i in range(self.total_points):
            x = int(self.x[i][0] * self.img_size[0])
            y = self.img_size[1] - int(self.x[i][1] * self.img_size[1])
            fill = (64, 64, 255) if self.y[i] == 1 else (255, 64, 64)
            outline = (128, 128, 255) if self.y[i] == 1 else (255, 128, 128)
            draw.ellipse(((x - 5, y - 5), (x + 5, y + 5)), outline=outline, fill=fill)
        #img.show()
        return img

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.graphic = QtWidgets.QLabel(self.centralwidget)
        self.graphic.setGeometry(QtCore.QRect(10, 60, 400, 400))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.graphic.sizePolicy().hasHeightForWidth())
        self.graphic.setSizePolicy(sizePolicy)
        self.graphic.setMinimumSize(QtCore.QSize(400, 400))
        self.graphic.setBaseSize(QtCore.QSize(400, 400))
        self.graphic.setObjectName("graphic")
        # self.img = QtGui.QPixmap()
        # self.graphic.setPixmap(self.img)
        self.parameters_group = QtWidgets.QGroupBox(self.centralwidget)
        self.parameters_group.setGeometry(QtCore.QRect(430, 60, 365, 269))
        self.parameters_group.setObjectName("parameters_group")
        self.activation_function_group = QtWidgets.QGroupBox(self.parameters_group)
        self.activation_function_group.setGeometry(QtCore.QRect(9, 40, 351, 71))
        self.activation_function_group.setObjectName("activation_function_group")
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(self.activation_function_group)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(9, 29, 331, 41))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.activation_function_box = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.activation_function_box.setContentsMargins(0, 0, 0, 0)
        self.activation_function_box.setObjectName("activation_function_box")
        self.step = QtWidgets.QRadioButton(self.horizontalLayoutWidget_2)
        self.step.setChecked(True)
        self.step.setObjectName("step")
        self.activation_function_box.addWidget(self.step)
        self.sigmoid = QtWidgets.QRadioButton(self.horizontalLayoutWidget_2)
        self.sigmoid.setObjectName("sigmoid")
        self.activation_function_box.addWidget(self.sigmoid)
        self.train_group = QtWidgets.QGroupBox(self.parameters_group)
        self.train_group.setGeometry(QtCore.QRect(9, 129, 351, 131))
        self.train_group.setObjectName("train_group")
        self.formLayoutWidget_2 = QtWidgets.QWidget(self.train_group)
        self.formLayoutWidget_2.setGeometry(QtCore.QRect(9, 39, 331, 91))
        self.formLayoutWidget_2.setObjectName("formLayoutWidget_2")
        self.train_form = QtWidgets.QFormLayout(self.formLayoutWidget_2)
        self.train_form.setContentsMargins(0, 0, 0, 0)
        self.train_form.setObjectName("train_form")
        self.learning_rate_label = QtWidgets.QLabel(self.formLayoutWidget_2)
        self.learning_rate_label.setObjectName("learning_rate_label")
        self.train_form.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.learning_rate_label)
        self.generations_label = QtWidgets.QLabel(self.formLayoutWidget_2)
        self.generations_label.setObjectName("generations_label")
        self.train_form.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.generations_label)
        self.learning_rate_spin = QtWidgets.QDoubleSpinBox(self.formLayoutWidget_2)
        self.learning_rate_spin.setDecimals(3)
        self.learning_rate_spin.setMaximum(1.0)
        self.learning_rate_spin.setSingleStep(0.005)
        self.learning_rate_spin.setProperty("value", 0.1)
        self.learning_rate_spin.setObjectName("learning_rate_spin")
        self.train_form.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.learning_rate_spin)
        self.generations_spin = QtWidgets.QSpinBox(self.formLayoutWidget_2)
        self.generations_spin.setMinimum(1)
        self.generations_spin.setMaximum(9999)
        self.generations_spin.setSingleStep(10)
        self.generations_spin.setProperty("value", 1)
        self.generations_spin.setObjectName("generations_spin")
        self.train_form.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.generations_spin)
        self.control_group = QtWidgets.QGroupBox(self.centralwidget)
        self.control_group.setGeometry(QtCore.QRect(430, 340, 365, 131))
        self.control_group.setObjectName("control_group")
        self.gridLayoutWidget_2 = QtWidgets.QWidget(self.control_group)
        self.gridLayoutWidget_2.setGeometry(QtCore.QRect(10, 30, 341, 42))
        self.gridLayoutWidget_2.setObjectName("gridLayoutWidget_2")
        self.control_grid_1 = QtWidgets.QGridLayout(self.gridLayoutWidget_2)
        self.control_grid_1.setContentsMargins(0, 0, 0, 0)
        self.control_grid_1.setObjectName("control_grid_1")
        self.randomize_button = QtWidgets.QPushButton(self.gridLayoutWidget_2)
        self.randomize_button.setObjectName("randomize_button")
        self.control_grid_1.addWidget(self.randomize_button, 0, 0, 1, 1)
        self.clear_button = QtWidgets.QPushButton(self.gridLayoutWidget_2)
        self.clear_button.setObjectName("clear_button")
        self.control_grid_1.addWidget(self.clear_button, 0, 1, 1, 1)
        self.verticalLayoutWidget = QtWidgets.QWidget(self.control_group)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(10, 209, 341, 51))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.data_group = QtWidgets.QGroupBox(self.centralwidget)
        self.data_group.setGeometry(QtCore.QRect(10, 470, 781, 121))
        self.data_group.setObjectName("data_group")
        self.gridLayoutWidget_3 = QtWidgets.QWidget(self.data_group)
        self.gridLayoutWidget_3.setGeometry(QtCore.QRect(10, 30, 381, 81))
        self.gridLayoutWidget_3.setObjectName("gridLayoutWidget_3")
        self.train_data_grid = QtWidgets.QGridLayout(self.gridLayoutWidget_3)
        self.train_data_grid.setContentsMargins(0, 0, 0, 0)
        self.train_data_grid.setObjectName("train_data_grid")
        self.episode = QtWidgets.QLabel(self.gridLayoutWidget_3)
        self.episode.setAlignment(QtCore.Qt.AlignCenter)
        self.episode.setObjectName("episode")
        self.train_data_grid.addWidget(self.episode, 1, 1, 1, 1)
        self.current_label = QtWidgets.QLabel(self.gridLayoutWidget_3)
        self.current_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.current_label.setObjectName("current_label")
        self.train_data_grid.addWidget(self.current_label, 0, 0, 1, 1)
        self.generation = QtWidgets.QLabel(self.gridLayoutWidget_3)
        self.generation.setAlignment(QtCore.Qt.AlignCenter)
        self.generation.setObjectName("generation")
        self.train_data_grid.addWidget(self.generation, 1, 3, 1, 1)
        self.loss = QtWidgets.QLabel(self.gridLayoutWidget_3)
        self.loss.setAlignment(QtCore.Qt.AlignCenter)
        self.loss.setObjectName("loss")
        self.train_data_grid.addWidget(self.loss, 0, 3, 1, 1)
        self.current = QtWidgets.QLabel(self.gridLayoutWidget_3)
        self.current.setAlignment(QtCore.Qt.AlignCenter)
        self.current.setObjectName("current")
        self.train_data_grid.addWidget(self.current, 0, 1, 1, 1)
        self.generation_label = QtWidgets.QLabel(self.gridLayoutWidget_3)
        self.generation_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.generation_label.setObjectName("generation_label")
        self.train_data_grid.addWidget(self.generation_label, 1, 2, 1, 1)
        self.episode_label = QtWidgets.QLabel(self.gridLayoutWidget_3)
        self.episode_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.episode_label.setObjectName("episode_label")
        self.train_data_grid.addWidget(self.episode_label, 1, 0, 1, 1)
        self.loss_label = QtWidgets.QLabel(self.gridLayoutWidget_3)
        self.loss_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.loss_label.setObjectName("loss_label")
        self.train_data_grid.addWidget(self.loss_label, 0, 2, 1, 1)
        self.model_group = QtWidgets.QGroupBox(self.data_group)
        self.model_group.setGeometry(QtCore.QRect(420, 40, 351, 71))
        self.model_group.setObjectName("model_group")
        self.b = QtWidgets.QLabel(self.model_group)
        self.b.setGeometry(QtCore.QRect(290, 40, 61, 23))
        self.b.setObjectName("b")
        self.w1_label = QtWidgets.QLabel(self.model_group)
        self.w1_label.setGeometry(QtCore.QRect(10, 40, 31, 23))
        self.w1_label.setObjectName("w1_label")
        self.w2_label = QtWidgets.QLabel(self.model_group)
        self.w2_label.setGeometry(QtCore.QRect(130, 40, 31, 23))
        self.w2_label.setObjectName("w2_label")
        self.blabel = QtWidgets.QLabel(self.model_group)
        self.blabel.setGeometry(QtCore.QRect(250, 40, 31, 23))
        self.blabel.setObjectName("blabel")
        self.w2 = QtWidgets.QLabel(self.model_group)
        self.w2.setGeometry(QtCore.QRect(170, 40, 61, 23))
        self.w2.setObjectName("w2")
        self.w1 = QtWidgets.QLabel(self.model_group)
        self.w1.setGeometry(QtCore.QRect(50, 40, 61, 23))
        self.w1.setObjectName("w1")
        self.title = QtWidgets.QLabel(self.centralwidget)
        self.title.setGeometry(QtCore.QRect(19, 10, 771, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.title.setFont(font)
        self.title.setAlignment(QtCore.Qt.AlignCenter)
        self.title.setObjectName("title")
        self.gridLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(440, 420, 341, 42))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.control_grid_2 = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.control_grid_2.setContentsMargins(0, 0, 0, 0)
        self.control_grid_2.setObjectName("control_grid_2")
        self.step_button = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.step_button.setObjectName("step_button")
        self.control_grid_2.addWidget(self.step_button, 0, 1, 1, 1)
        self.train_button = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.train_button.setObjectName("train_button")
        self.control_grid_2.addWidget(self.train_button, 0, 0, 1, 1)
        self.stop_button = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.stop_button.setObjectName("stop_button")
        self.control_grid_2.addWidget(self.stop_button, 0, 2, 1, 1)
        self.stop_button.setEnabled(False)
        self.step_button.setEnabled(False)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "VisualPerceptron 1.0"))
        self.parameters_group.setTitle(_translate("MainWindow", "Parâmetros"))
        self.activation_function_group.setTitle(_translate("MainWindow", "Função de ativação"))
        self.step.setText(_translate("MainWindow", "Degrau"))
        self.sigmoid.setText(_translate("MainWindow", "Sigmoide"))
        self.train_group.setTitle(_translate("MainWindow", "Treinamento"))
        self.learning_rate_label.setText(_translate("MainWindow", "Taxa de Aprendizagem"))
        self.generations_label.setText(_translate("MainWindow", "Gerações"))
        self.control_group.setTitle(_translate("MainWindow", "Controle"))
        self.randomize_button.setText(_translate("MainWindow", "Pesos aleatórios"))
        self.clear_button.setText(_translate("MainWindow", "Limpar"))
        self.data_group.setTitle(_translate("MainWindow", "Indicadores"))
        self.episode.setText(_translate("MainWindow", "1/100"))
        self.current_label.setText(_translate("MainWindow", "Ponto Atual:"))
        self.generation.setText(_translate("MainWindow", "1/8"))
        self.loss.setText(_translate("MainWindow", "0.00"))
        self.current.setText(_translate("MainWindow", "(x, y)"))
        self.generation_label.setText(_translate("MainWindow", "Geração:"))
        self.episode_label.setText(_translate("MainWindow", "Episódio:"))
        self.loss_label.setText(_translate("MainWindow", "Perda:"))
        self.model_group.setTitle(_translate("MainWindow", "Modelo Atual"))
        self.b.setText(_translate("MainWindow", "0.632"))
        self.w1_label.setText(_translate("MainWindow", "w1:"))
        self.w2_label.setText(_translate("MainWindow", "w2:"))
        self.blabel.setText(_translate("MainWindow", "b:"))
        self.w2.setText(_translate("MainWindow", "0.632"))
        self.w1.setText(_translate("MainWindow", "0.632"))
        self.title.setText(_translate("MainWindow", "VisualPerceptron 1.0                            by Gustavo Denobi"))
        self.step_button.setText(_translate("MainWindow", "Passo"))
        self.train_button.setText(_translate("MainWindow", "Treinar"))
        self.stop_button.setText(_translate("MainWindow", "Parar"))


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):

    def __init__(self):
        """ Initialization
        Parameters
        ----------
        """
        # Base class
        QtWidgets.QMainWindow.__init__(self)

        self.ia = Perceptron()

        # Initialize the UI widgets
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setupBindings()
        self.refreshView()

    def run(self):
        self.ia.stop = False
        if (self.ia.check_sizes()):
            while(not self.ia.stop):
                if (self.ia.current_index == self.ia.total_points):
                    self.ia.current_index = 0
                    self.ia.current_generation += 1
                    self.refreshView()
                if (self.ia.current_generation == self.ia.generations + 1):
                    self.ia.stop = True
                    self.ia.current_generation = 0
                if (not self.ia.stop):
                    i = self.ia.current_index
                    self.ia.current_total_error = self.ia.error_function()
                    mult, y_pred = self.ia.forward_pass(self.ia.x[i][0], self.ia.x[i][1])
                    self.ia.train(self.ia.x[i], self.ia.y[i], y_pred, mult)
                self.refreshLabels()
                self.ia.current_index += 1



    def step(self):
        pass

    def stop(self):
        self.ia.stop = True
        self.refreshLabels()

    def randomize(self):
        self.ia.randomize_w()
        self.refreshView()

    def clear(self):
        self.ia.clear()
        self.refreshView()

    def toggle_activation_func(self):
        if(self.ia.activation_function == "step"):
            self.ia.activation_function = "sigmoid"
        elif(self.ia.activation_function == "sigmoid"):
            self.ia.activation_function = "step"
        self.refreshView()

    def update_learning_rate(self):
        self.ia.learning_rate = float(self.ui.learning_rate_spin.value())
        self.refreshLabels()

    def update_generations(self):
        self.ia.generations = int(self.ui.generations_spin.value())
        self.refreshLabels()

    def add_point(self, event):
        x = event.pos().x()
        y = -event.pos().y() + self.ia.img_size[1]
        truth = 1 if (event.button() == QtCore.Qt.LeftButton) else 0
        self.ia.add_point(x, y, truth)
        self.refreshView()

    def setupBindings(self):
        self.ui.clear_button.clicked.connect(self.clear)
        self.ui.train_button.clicked.connect(self.run)
        self.ui.step_button.clicked.connect(self.step)
        self.ui.stop_button.clicked.connect(self.stop)
        self.ui.randomize_button.clicked.connect(self.randomize)
        self.ui.clear_button.clicked.connect(self.clear)
        self.ui.sigmoid.toggled.connect(self.toggle_activation_func)
        self.ui.learning_rate_spin.valueChanged.connect(self.update_learning_rate)
        self.ui.generations_spin.valueChanged.connect(self.update_generations)
        self.ui.graphic.mousePressEvent = self.add_point

    def refreshView(self):
        self.refreshLabels()
        self.drawImage()

    def refreshLabels(self):
        current_index = self.ia.current_index
        if (self.ia.total_points > 0):
            self.ui.current.setText("({:5.2f}, {:5.2f})".format(self.ia.x[current_index][0], self.ia.x[current_index][1]))
        else:
            self.ui.current.setText("(x, y)")
        self.ui.loss.setText("{:5.2f}".format(self.ia.current_total_error))
        self.ui.episode.setText("{}/{}".format(current_index, self.ia.total_points))
        self.ui.generation.setText("{}/{}".format(self.ia.current_generation, self.ia.generations))
        self.ui.w1.setText("{:5.2f}".format(self.ia.weights[0]))
        self.ui.w2.setText("{:5.2f}".format(self.ia.weights[1]))
        self.ui.b.setText("{:5.2f}".format(self.ia.weights[2]))
        self.ui.episode.repaint()
        self.update()

    def drawImage(self):

        self.ia.img = self.ia.generate_img()
        img = self.ia.img.convert("RGB")
        data = img.tobytes("raw", "RGB")
        img = QtGui.QImage(data, img.size[0], img.size[1], QtGui.QImage.Format_RGB888)
        img = QtGui.QPixmap.fromImage(img)

        #img = QtGui.QPixmap.fromImage(QtGui.QImage(ImageQt.ImageQt(self.ia.img)))
        self.ui.graphic.setPixmap(img)
        self.ui.graphic.repaint()

def main():
    app = QtWidgets.QApplication(sys.argv)
    application = MainWindow()
    application.show()
    app.exec_()

def except_hook(cls, exception, traceback):
    sys.__excepthook__(cls, exception, traceback)


if __name__ == '__main__':
    sys.excepthook = except_hook
    main()