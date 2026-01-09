import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QFileDialog,
                             QTextEdit, QProgressBar, QMessageBox, QSlider)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QTimer

# 引入特征提取
try:
    from core.feature_extractor import extract_features
except ImportError:
    QMessageBox.critical(None, "错误", "找不到 core/feature_extractor.py！")
    sys.exit(1)


def cv_imread(file_path):
    try:
        cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
        return cv_img
    except Exception:
        return None


class FruitAutoSystem(QMainWindow):
    def __init__(self):
        super().__init__()

        # === 配置参数 ===
        self.svm_path = "models/svm_model.xml"
        self.cnn_path = "models/cnn_model.onnx"

        # 标签顺序
        self.classes = [
            "Apple_Overripe", "Apple_Ripe", "Apple_Unripe",
            "Orange_Overripe", "Orange_Ripe", "Orange_Unripe"
        ]
        # 设置模型投票权重
        self.weight_cnn = 0.85
        self.weight_svm = 0.15
        self.threshold = 0.70

        # 摄像头相关
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.is_camera_active = False
        self.frame_count = 0  # 计数器

        self.current_frame = None

        self.initUI()
        self.load_models()

    def initUI(self):
        self.setWindowTitle('水果分级系统 (全自动实时检测)')
        self.setGeometry(100, 100, 1050, 680)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # ===========================
        # 左侧：显示区
        # ===========================
        left_layout = QVBoxLayout()

        self.lbl_image = QLabel("摄像头关闭")
        self.lbl_image.setAlignment(Qt.AlignCenter)
        self.lbl_image.setStyleSheet(
            "border: 2px dashed #aaa; background-color: #f0f0f0; font-size: 20px; color: #aaa;")
        self.lbl_image.setFixedSize(500, 500)

        # 按钮组
        btn_layout = QHBoxLayout()
        self.btn_load = QPushButton("📂 识别本地图片")  # 改了个名，暗示点这个也会自动识别
        self.btn_load.clicked.connect(self.open_file)
        self.btn_load.setFixedHeight(50)
        self.btn_load.setFont(QFont("Arial", 12))

        self.btn_cam = QPushButton("📷 打开摄像头 (自动识别)")
        self.btn_cam.clicked.connect(self.toggle_camera)
        self.btn_cam.setFixedHeight(50)
        self.btn_cam.setFont(QFont("Arial", 12, QFont.Bold))
        self.btn_cam.setStyleSheet("background-color: #007AFF; color: white;")

        btn_layout.addWidget(self.btn_load)
        btn_layout.addWidget(self.btn_cam)

        left_layout.addWidget(self.lbl_image)
        left_layout.addLayout(btn_layout)

        # ===========================
        # 右侧：结果面板
        # ===========================
        right_layout = QVBoxLayout()

        self.txt_features = QTextEdit()
        self.txt_features.setReadOnly(True)
        self.txt_features.setMaximumHeight(100)
        self.txt_features.setPlaceholderText("实时特征数据...")

        thresh_layout = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(int(self.threshold * 100))
        self.slider.valueChanged.connect(self.update_thresh_label)

        self.lbl_thresh = QLabel(f"{self.threshold:.2f}")
        self.lbl_thresh.setStyleSheet("color: blue; font-weight: bold;")

        thresh_layout.addWidget(QLabel("灵敏度阈值:"))
        thresh_layout.addWidget(self.slider)
        thresh_layout.addWidget(self.lbl_thresh)

        # 结果显示做得大一点
        self.lbl_result = QLabel("准备就绪")
        self.lbl_result.setAlignment(Qt.AlignCenter)
        self.lbl_result.setFont(QFont("Arial", 28, QFont.Bold))
        self.lbl_result.setStyleSheet("border: 2px solid #ddd; padding: 20px; color: #ccc; border-radius: 10px;")

        self.pbar = QProgressBar()
        self.pbar.setRange(0, 100)
        self.pbar.setValue(0)
        self.pbar.setFixedHeight(20)
        self.pbar.setStyleSheet("QProgressBar::chunk { background-color: #007AFF; }")

        right_layout.addWidget(QLabel("📊 实时特征:"))
        right_layout.addWidget(self.txt_features)
        right_layout.addSpacing(15)
        right_layout.addLayout(thresh_layout)
        right_layout.addSpacing(30)
        right_layout.addWidget(QLabel("🍎 识别结果:"))
        right_layout.addWidget(self.lbl_result)
        right_layout.addSpacing(15)
        right_layout.addWidget(QLabel("📈 置信度:"))
        right_layout.addWidget(self.pbar)
        right_layout.addStretch()

        main_layout.addLayout(left_layout, 6)
        main_layout.addLayout(right_layout, 4)

    def load_models(self):
        try:
            self.svm = cv2.ml.SVM_load(self.svm_path)
            self.net = cv2.dnn.readNetFromONNX(self.cnn_path)
            print("✅ 模型加载成功")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"模型加载失败: {e}")

    def toggle_camera(self):
        if not self.is_camera_active:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                QMessageBox.warning(self, "错误", "无法打开摄像头！")
                return

            self.timer.start(30)  # 30ms 刷新
            self.btn_cam.setText("🚫 关闭摄像头")
            self.btn_cam.setStyleSheet("background-color: #d9534f; color: white;")
            self.is_camera_active = True
            self.btn_load.setEnabled(False)
            self.lbl_result.setText("正在识别...")
            self.lbl_result.setStyleSheet("border: 2px solid #007AFF; color: #007AFF;")
        else:
            self.timer.stop()
            self.cap.release()
            self.lbl_image.setPixmap(QPixmap())
            self.lbl_image.setText("摄像头已关闭")
            self.btn_cam.setText("📷 打开摄像头 (自动识别)")
            self.btn_cam.setStyleSheet("background-color: #007AFF; color: white;")
            self.is_camera_active = False
            self.btn_load.setEnabled(True)
            self.lbl_result.setText("准备就绪")
            self.lbl_result.setStyleSheet("border: 2px solid #ddd; color: #ccc;")
            self.pbar.setValue(0)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame
            self.display_image(frame)

            # === 核心：只要有画面，就自动测 ===
            self.frame_count += 1
            if self.frame_count % 5 == 0:  # 每 5 帧测一次 (平滑不卡顿)
                self.run_fusion_detection(manual=False)

    def open_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, '打开图片', './dataset/test', "Image files (*.jpg *.png)")
        if fname:
            img = cv_imread(fname)
            if img is not None:
                self.current_frame = img
                self.display_image(img)
                # 加载图片后，立即自动跑一次
                self.run_fusion_detection(manual=True)
            else:
                QMessageBox.warning(self, "错误", "读取图片失败")

    def display_image(self, img_bgr):
        rgb_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.lbl_image.setPixmap(QPixmap.fromImage(qt_img).scaled(500, 500, Qt.KeepAspectRatio))

    def update_thresh_label(self, val):
        self.threshold = val / 100.0
        self.lbl_thresh.setText(f"{self.threshold:.2f}")

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def run_fusion_detection(self, manual=True):
        if self.current_frame is None: return

        img = self.current_frame.copy()

        # 1. SVM
        svm_probs = np.zeros(6)
        svm_label = -1
        feats = extract_features(image_data=img)

        if feats is not None:
            # 只在手动或低频时更新文本，防止太闪
            if manual or self.frame_count % 10 == 0:
                f_str = f"RGB: {feats[0]:.0f},{feats[1]:.0f},{feats[2]:.0f}\n"
                f_str += f"HSV: {feats[3]:.0f},{feats[4]:.0f},{feats[5]:.0f}\n"
                self.txt_features.setText(f_str)

            svm_in = np.matrix(feats, dtype=np.float32)
            _, resp = self.svm.predict(svm_in)
            svm_label = int(resp[0, 0])
            svm_probs[svm_label] = 1.0

            # 2. CNN
        blob = cv2.dnn.blobFromImage(img, 1.0 / 255.0, (224, 224), (0, 0, 0), swapRB=True, crop=False)
        self.net.setInput(blob)
        cnn_out = self.net.forward()
        cnn_probs = self.softmax(cnn_out[0])

        # 3. 融合
        final_probs = (cnn_probs * self.weight_cnn) + (svm_probs * self.weight_svm)
        max_idx = np.argmax(final_probs)
        confidence = final_probs[max_idx]

        self.pbar.setValue(int(confidence * 100))

        if confidence >= self.threshold:
            raw_cls = self.classes[max_idx]
            fruit, grade = raw_cls.split("_")
            cn_map = {"Apple": "苹果", "Orange": "橘子",
                      "Unripe": "未成熟", "Ripe": "成熟", "Overripe": "过成熟"}

            # 颜色逻辑：熟=绿，未熟=蓝，过熟=红
            color_style = "#28a745"
            if grade == "Overripe": color_style = "#d9534f"
            if grade == "Unripe": color_style = "#17a2b8"

            final_text = f"{cn_map.get(fruit, fruit)}\n{cn_map.get(grade, grade)}"
            self.lbl_result.setText(final_text)
            self.lbl_result.setStyleSheet(
                f"border: 3px solid {color_style}; color: {color_style}; border-radius: 10px;")
        else:
            self.lbl_result.setText("无法识别")
            self.lbl_result.setStyleSheet("border: 3px solid #ccc; color: #ccc; border-radius: 10px;")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = FruitAutoSystem()
    ex.show()
    sys.exit(app.exec_())