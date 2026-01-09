import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QFileDialog,
                             QTextEdit, QProgressBar, QMessageBox, QSlider)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt

# 引入特征提取 (仅用于界面展示，不参与预测)
try:
    from core.feature_extractor import extract_features
except ImportError:
    # 如果找不到 core 模块，为了防止报错，定义一个假的函数
    def extract_features(image_data):
        return None


class FruitGradingAppCNN(QMainWindow):
    def __init__(self):
        super().__init__()

        # === 1. 配置参数 ===
        self.cnn_path = "models/cnn_model.onnx"
        self.classes = [
            "Apple_Overripe", "Apple_Ripe", "Apple_Unripe",
            "Orange_Overripe", "Orange_Ripe", "Orange_Unripe"
        ]

        # 默认阈值
        self.threshold = 0.70

        # 缓存数据
        self.last_probs = None

        self.initUI()
        self.load_models()

    def initUI(self):
        self.setWindowTitle('水果成熟度分级系统 (纯 CNN 模式)')
        self.setGeometry(100, 100, 950, 650)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # ===========================
        # 左侧：图像显示
        # ===========================
        left_layout = QVBoxLayout()

        self.lbl_image = QLabel("请加载图片")
        self.lbl_image.setAlignment(Qt.AlignCenter)
        self.lbl_image.setStyleSheet("border: 2px dashed #aaa; background-color: #f0f0f0;")
        self.lbl_image.setFixedSize(400, 400)

        btn_load = QPushButton("📂 加载图片")
        btn_load.setFixedSize(120, 40)
        btn_load.clicked.connect(self.open_image)

        btn_detect = QPushButton("🔍 CNN 识别")
        btn_detect.setFixedSize(120, 40)
        btn_detect.setStyleSheet("background-color: #FF9500; color: white; font-weight: bold;")
        btn_detect.clicked.connect(self.run_detection)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(btn_load)
        btn_layout.addWidget(btn_detect)

        left_layout.addWidget(self.lbl_image)
        left_layout.addLayout(btn_layout)

        # ===========================
        # 右侧：结果显示
        # ===========================
        right_layout = QVBoxLayout()

        # 1. 特征数据 (仅展示)
        lbl_feat_title = QLabel("📊 图像特征 (仅展示):")
        lbl_feat_title.setFont(QFont("Arial", 10, QFont.Bold))
        self.txt_features = QTextEdit()
        self.txt_features.setReadOnly(True)
        self.txt_features.setMaximumHeight(120)

        # 2. 阈值控制
        lbl_thresh_title = QLabel("🎚️ 判定阈值设置:")
        lbl_thresh_title.setFont(QFont("Arial", 10, QFont.Bold))

        thresh_layout = QHBoxLayout()
        self.slider_thresh = QSlider(Qt.Horizontal)
        self.slider_thresh.setRange(0, 100)
        self.slider_thresh.setValue(int(self.threshold * 100))
        self.slider_thresh.valueChanged.connect(self.update_threshold)

        self.lbl_thresh_val = QLabel(f"{self.threshold:.2f}")
        self.lbl_thresh_val.setFixedWidth(40)
        self.lbl_thresh_val.setFont(QFont("Arial", 10, QFont.Bold))
        self.lbl_thresh_val.setStyleSheet("color: blue;")

        thresh_layout.addWidget(self.slider_thresh)
        thresh_layout.addWidget(self.lbl_thresh_val)

        # 3. 结果显示
        lbl_res_title = QLabel("🍎 CNN 分级结论:")
        lbl_res_title.setFont(QFont("Arial", 12, QFont.Bold))

        self.lbl_result = QLabel("等待检测...")
        self.lbl_result.setFont(QFont("Arial", 16, QFont.Bold))
        self.lbl_result.setStyleSheet("color: #666; padding: 10px; border: 1px solid #ddd;")
        self.lbl_result.setAlignment(Qt.AlignCenter)

        # 4. 置信度条
        lbl_conf_title = QLabel("📈 置信度 (Confidence):")
        self.pbar_conf = QProgressBar()
        self.pbar_conf.setRange(0, 100)
        self.pbar_conf.setValue(0)
        self.pbar_conf.setStyleSheet("QProgressBar::chunk { background-color: #FF9500; }")

        right_layout.addWidget(lbl_feat_title)
        right_layout.addWidget(self.txt_features)
        right_layout.addSpacing(15)
        right_layout.addWidget(lbl_thresh_title)
        right_layout.addLayout(thresh_layout)
        right_layout.addSpacing(15)
        right_layout.addWidget(lbl_res_title)
        right_layout.addWidget(self.lbl_result)
        right_layout.addSpacing(10)
        right_layout.addWidget(lbl_conf_title)
        right_layout.addWidget(self.pbar_conf)
        right_layout.addStretch()

        main_layout.addLayout(left_layout, 2)
        main_layout.addLayout(right_layout, 3)

        self.current_image_path = None
        self.current_cv_image = None

    def load_models(self):
        try:
            print(f"Loading CNN from {self.cnn_path}...")
            self.net = cv2.dnn.readNetFromONNX(self.cnn_path)
            print("✅ CNN 模型加载成功！")
        except Exception as e:
            QMessageBox.critical(self, "模型加载失败", f"请检查 models 文件夹。\n错误: {str(e)}")

    def update_threshold(self, value):
        self.threshold = value / 100.0
        self.lbl_thresh_val.setText(f"{self.threshold:.2f}")
        # 实时刷新结果
        if self.last_probs is not None:
            self.show_final_decision(self.last_probs)

    def open_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, '打开图片', './dataset/test', "Image files (*.jpg *.png)")
        if fname:
            self.current_image_path = fname
            self.current_cv_image = cv2.imread(fname)

            # 显示图片
            rgb_img = cv2.cvtColor(self.current_cv_image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_img.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg).scaled(400, 400, Qt.KeepAspectRatio)
            self.lbl_image.setPixmap(pixmap)

            # 重置状态
            self.lbl_result.setText("就绪")
            self.lbl_result.setStyleSheet("color: black;")
            self.txt_features.clear()
            self.pbar_conf.setValue(0)
            self.last_probs = None

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def run_detection(self):
        if self.current_cv_image is None: return
        img = self.current_cv_image  # BGR格式

        # 1. 显示特征 (UI展示用)
        feats = extract_features(image_data=img)
        if feats is not None:
            feat_str = "RGB: {:.0f}, {:.0f}, {:.0f}\n".format(*feats[0:3])
            feat_str += "HSV: {:.0f}, {:.0f}, {:.0f}\n".format(*feats[3:6])
            feat_str += "纹理: {:.2f}, {:.2f}, {:.2f}".format(*feats[9:12])
            self.txt_features.setText(feat_str)
        else:
            self.txt_features.setText("无法提取特征 (模块缺失?)")

        # 2. CNN 预测 (核心)
        # 必须 SwapRB=True (BGR->RGB)
        # 必须 1.0/255.0 (归一化到0-1)
        blob = cv2.dnn.blobFromImage(img, 1.0 / 255.0, (224, 224), (0, 0, 0), swapRB=True, crop=False)

        self.net.setInput(blob)
        cnn_out = self.net.forward()
        cnn_probs = self.softmax(cnn_out[0])

        print(f"CNN Probs: {np.round(cnn_probs, 2)}")

        # 3. 显示结果
        self.last_probs = cnn_probs
        self.show_final_decision(cnn_probs)

    def show_final_decision(self, probs):
        max_idx = np.argmax(probs)
        confidence = probs[max_idx]

        self.pbar_conf.setValue(int(confidence * 100))

        if confidence >= self.threshold:
            class_name = self.classes[max_idx]
            fruit, grade = class_name.split("_")
            cn_map = {"Apple": "苹果", "Orange": "橘子/橙子",
                      "Unripe": "未成熟", "Ripe": "成熟", "Overripe": "过成熟"}
            res_text = f"{cn_map.get(fruit, fruit)} - {cn_map.get(grade, grade)}"

            self.lbl_result.setText(res_text)
            self.lbl_result.setStyleSheet(
                "color: #FF9500; font-weight: bold; font-size: 20px; border: 2px solid #FF9500;")
        else:
            self.lbl_result.setText("无法准确分级\n(置信度不足)")
            self.lbl_result.setStyleSheet("color: red; font-weight: bold; border: 2px solid red;")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = FruitGradingAppCNN()
    ex.show()
    sys.exit(app.exec_())