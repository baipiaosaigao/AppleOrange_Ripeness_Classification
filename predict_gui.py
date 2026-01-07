import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QFileDialog,
                             QTextEdit, QProgressBar, QMessageBox, QSlider)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt

# 引入特征提取 (SVM 用)
from core.feature_extractor import extract_features


class FruitGradingApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # === 1. 配置参数 ===
        self.svm_path = "models/svm_model.xml"
        self.cnn_path = "models/cnn_model.onnx"
        self.classes = [
            "Apple_Unripe", "Apple_Ripe", "Apple_Overripe",
            "Orange_Unripe", "Orange_Ripe", "Orange_Overripe"
        ]

        # 融合权重 (CNN权重0.6, SVM权重0.4)
        self.weight_cnn = 0.6
        self.weight_svm = 0.4

        # 默认阈值
        self.threshold = 0.70

        # 缓存上一次的结果用于滑块实时刷新
        self.last_final_probs = None

        self.initUI()
        self.load_models()

    def initUI(self):
        self.setWindowTitle('水果成熟度分级系统 (SVM + CNN 融合决策)')
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

        btn_detect = QPushButton("🔍 融合识别")
        btn_detect.setFixedSize(120, 40)
        btn_detect.setStyleSheet("background-color: #007AFF; color: white; font-weight: bold;")
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

        # 1. 特征数据
        lbl_feat_title = QLabel("📊 图像特征 (SVM输入):")
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
        lbl_res_title = QLabel("🍎 融合分级结论:")
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
        self.pbar_conf.setStyleSheet("QProgressBar::chunk { background-color: #007AFF; }")

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
            # 加载 SVM
            print(f"Loading SVM from {self.svm_path}...")
            self.svm = cv2.ml.SVM_load(self.svm_path)

            # 加载 CNN
            print(f"Loading CNN from {self.cnn_path}...")
            self.net = cv2.dnn.readNetFromONNX(self.cnn_path)
            print("✅ 模型加载成功！")
        except Exception as e:
            QMessageBox.critical(self, "模型加载失败", f"请检查 models 文件夹。\n错误: {str(e)}")

    def update_threshold(self, value):
        self.threshold = value / 100.0
        self.lbl_thresh_val.setText(f"{self.threshold:.2f}")
        if self.last_final_probs is not None:
            self.show_final_decision(self.last_final_probs)

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
            self.last_final_probs = None

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def run_detection(self):
        if self.current_cv_image is None: return
        img = self.current_cv_image  # 这里是 BGR 格式

        # ==========================================
        # 1. SVM 处理流程 (预处理：直接用 BGR，不需要缩放)
        # ==========================================
        svm_probs = np.zeros(6)
        svm_label = -1

        feats = extract_features(image_data=img)
        if feats is not None:
            # 显示特征
            feat_str = "RGB: {:.1f}, {:.1f}, {:.1f} | ".format(*feats[0:3])
            feat_str += "HSV: {:.1f}, {:.1f}, {:.1f}\n".format(*feats[3:6])
            feat_str += "HSV_Std: {:.2f}, {:.2f}, {:.2f}\n".format(*feats[6:9])
            feat_str += "Texture: {:.2f}, {:.2f}, {:.2f}".format(*feats[9:12])
            self.txt_features.setText(feat_str)

            # 预测 (SVM必须输入 float32 矩阵)
            svm_input = np.matrix(feats, dtype=np.float32)
            ret, svm_response = self.svm.predict(svm_input)
            svm_label = int(svm_response[0, 0])

            # 模拟 SVM 概率 (命中为1，未命中为0)
            for i in range(6):
                svm_probs[i] = 1.0 if i == svm_label else 0.0
        else:
            self.txt_features.setText("特征提取失败")

        # ==========================================
        # 2. CNN 处理流程 (预处理：SwapRB, 0-1缩放)
        # ==========================================
        # 关键点：1.0/255.0 对应 ToTensor()
        # 关键点：swapRB=True 对应 OpenCV(BGR) -> PyTorch(RGB)
        # 关键点：mean=(0,0,0) 因为我们在训练时注释掉了 Normalize
        blob = cv2.dnn.blobFromImage(img, 1.0 / 255.0, (224, 224), (0, 0, 0), swapRB=True, crop=False)

        self.net.setInput(blob)
        cnn_out = self.net.forward()
        cnn_probs = self.softmax(cnn_out[0])

        print(f"SVM 预测: {self.classes[svm_label] if svm_label >= 0 else 'None'}")
        print(f"CNN 分布: {np.round(cnn_probs, 2)}")

        # ==========================================
        # 3. 融合决策
        # ==========================================
        final_probs = (cnn_probs * self.weight_cnn) + (svm_probs * self.weight_svm)

        # 存下来用于滑块调整
        self.last_final_probs = final_probs
        self.show_final_decision(final_probs)

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
            self.lbl_result.setStyleSheet("color: green; font-weight: bold; font-size: 20px; border: 2px solid green;")
        else:
            self.lbl_result.setText("无法准确分级\n(置信度不足)")
            self.lbl_result.setStyleSheet("color: red; font-weight: bold; border: 2px solid red;")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = FruitGradingApp()
    ex.show()
    sys.exit(app.exec_())