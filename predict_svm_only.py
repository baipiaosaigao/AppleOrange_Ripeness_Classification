import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QFileDialog,
                             QTextEdit, QProgressBar, QMessageBox, QSlider)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt

# 引入特征提取
try:
    from core.feature_extractor import extract_features
except ImportError:
    QMessageBox.critical(None, "错误", "找不到 core/feature_extractor.py，请检查项目结构！")
    sys.exit(1)


# ==========================================
# 关键修复：支持中文路径读取的函数
# ==========================================
def cv_imread(file_path):
    """能够读取中文路径图片的辅助函数"""
    try:
        # np.fromfile 读取二进制，cv2.imdecode 解码
        cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
        return cv_img
    except Exception as e:
        print(f"读取图片失败: {e}")
        return None


class FruitGradingAppSVM(QMainWindow):
    def __init__(self):
        super().__init__()

        # === 1. 配置参数 ===
        self.svm_path = "models/svm_model.xml"

        # === 关键修正：必须按字母顺序排列 ===
        # 对应: 0, 1, 2, 3, 4, 5
        self.classes = [
            "Apple_Overripe", "Apple_Ripe", "Apple_Unripe",
            "Orange_Overripe", "Orange_Ripe", "Orange_Unripe"
        ]

        # 默认阈值
        self.threshold = 0.70

        # 缓存数据
        self.last_confidence = 0.0
        self.last_label_idx = -1

        self.initUI()
        self.load_models()

    def initUI(self):
        self.setWindowTitle('水果成熟度分级系统 (纯 SVM 模式)')
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

        btn_detect = QPushButton("🔍 SVM 识别")
        btn_detect.setFixedSize(120, 40)
        btn_detect.setStyleSheet("background-color: #8E44AD; color: white; font-weight: bold;")  # 紫色代表SVM
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

        # 1. 特征数据 (SVM 的核心输入)
        lbl_feat_title = QLabel("📊 提取的 12 维特征 (SVM 输入):")
        lbl_feat_title.setFont(QFont("Arial", 10, QFont.Bold))
        self.txt_features = QTextEdit()
        self.txt_features.setReadOnly(True)
        self.txt_features.setMaximumHeight(120)
        self.txt_features.setPlaceholderText("这里将显示 SVM 所需的颜色和纹理特征...")

        # 2. 阈值控制 (对硬分类器 SVM 来说主要是演示用)
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
        lbl_res_title = QLabel("🍎 SVM 分级结论:")
        lbl_res_title.setFont(QFont("Arial", 12, QFont.Bold))

        self.lbl_result = QLabel("等待检测...")
        self.lbl_result.setFont(QFont("Arial", 16, QFont.Bold))
        self.lbl_result.setStyleSheet("color: #666; padding: 10px; border: 1px solid #ddd;")
        self.lbl_result.setAlignment(Qt.AlignCenter)

        # 4. 置信度条
        lbl_conf_title = QLabel("📈 置信度 (SVM 为硬分类，默认 100%):")
        self.pbar_conf = QProgressBar()
        self.pbar_conf.setRange(0, 100)
        self.pbar_conf.setValue(0)
        self.pbar_conf.setStyleSheet("QProgressBar::chunk { background-color: #8E44AD; }")

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
            print(f"Loading SVM from {self.svm_path}...")
            self.svm = cv2.ml.SVM_load(self.svm_path)
            print("✅ SVM 模型加载成功！")
        except Exception as e:
            QMessageBox.critical(self, "模型加载失败", f"请检查 models 文件夹下是否有 svm_model.xml。\n错误: {str(e)}")

    def update_threshold(self, value):
        self.threshold = value / 100.0
        self.lbl_thresh_val.setText(f"{self.threshold:.2f}")
        # 实时刷新结果
        if self.last_label_idx >= 0:
            self.show_final_decision(self.last_confidence, self.last_label_idx)

    def open_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, '打开图片', './dataset/test', "Image files (*.jpg *.png)")
        if fname:
            self.current_image_path = fname

            # === 使用修复后的中文读取函数 ===
            self.current_cv_image = cv_imread(fname)

            if self.current_cv_image is None:
                QMessageBox.warning(self, "警告", "图片读取失败，请检查路径或文件完整性。")
                return

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
            self.last_label_idx = -1

    def run_detection(self):
        if self.current_cv_image is None: return
        img = self.current_cv_image  # BGR格式，正是SVM feature_extractor 需要的

        # 1. 提取特征
        feats = extract_features(image_data=img)
        if feats is not None:
            # 显示特征数值
            feat_str = "RGB均值: {:.1f}, {:.1f}, {:.1f}\n".format(*feats[0:3])
            feat_str += "HSV均值: {:.1f}, {:.1f}, {:.1f}\n".format(*feats[3:6])
            feat_str += "HSV标准差: {:.2f}, {:.2f}, {:.2f}\n".format(*feats[6:9])
            feat_str += "纹理(Con/Cor/Ene): {:.2f}, {:.2f}, {:.2f}".format(*feats[9:12])
            self.txt_features.setText(feat_str)

            # 2. SVM 预测
            # OpenCV SVM 要求输入必须是 float32 类型的 2D 矩阵
            svm_input = np.matrix(feats, dtype=np.float32)

            # predict 返回: (ret, results)
            # results 是一个 N x 1 的数组，存放类别索引
            ret, svm_response = self.svm.predict(svm_input)
            label_idx = int(svm_response[0, 0])

            # 3. 设置结果
            # 因为是硬分类，我们假设置信度为 1.0 (100%)
            confidence = 1.0

            print(f"SVM 预测类别索引: {label_idx} -> {self.classes[label_idx]}")

            # 保存状态并显示
            self.last_confidence = confidence
            self.last_label_idx = label_idx
            self.show_final_decision(confidence, label_idx)

        else:
            self.txt_features.setText("特征提取失败")
            QMessageBox.warning(self, "错误", "无法提取图像特征，可能是图片过小或格式不支持。")

    def show_final_decision(self, confidence, idx):
        self.pbar_conf.setValue(int(confidence * 100))

        # 如果置信度大于阈值 (对于SVM，只要阈值不大于1.0，永远通过)
        if confidence >= self.threshold:
            class_name = self.classes[idx]

            # 简单的字符串分割和汉化
            try:
                fruit, grade = class_name.split("_")
                cn_map = {"Apple": "苹果", "Orange": "橘子/橙子",
                          "Unripe": "未成熟", "Ripe": "成熟", "Overripe": "过成熟"}
                res_text = f"{cn_map.get(fruit, fruit)} - {cn_map.get(grade, grade)}"
            except:
                res_text = class_name  # 防止名字格式不对报错

            self.lbl_result.setText(res_text)
            self.lbl_result.setStyleSheet(
                "color: #8E44AD; font-weight: bold; font-size: 20px; border: 2px solid #8E44AD;")
        else:
            self.lbl_result.setText("无法准确分级\n(人为调高了阈值)")
            self.lbl_result.setStyleSheet("color: red; font-weight: bold; border: 2px solid red;")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = FruitGradingAppSVM()
    ex.show()
    sys.exit(app.exec_())