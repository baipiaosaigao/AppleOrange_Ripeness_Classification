# 文件: train_svm.py (OpenCV XML 专用版)
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from core.feature_extractor import extract_features

# 1. 标签映射 (6分类)
# 修改后 (严格对应字母顺序)
LABEL_MAP = {
    "apple_overripe": 0,
    "apple_ripe": 1,
    "apple_unripe": 2,
    "orange_overripe": 3,
    "orange_ripe": 4,
    "orange_unripe": 5
}


def train_svm_xml():
    print("=== [OpenCV版] 开始提取特征并训练 SVM ===")
    data_dir = './dataset/train'
    X = []
    y = []

    # 1. 读取数据
    for label_name, label_idx in LABEL_MAP.items():
        folder_path = os.path.join(data_dir, label_name)
        if not os.path.exists(folder_path): continue

        print(f"正在处理: {label_name} ...")
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            features = extract_features(image_path=img_path)
            if features is not None:
                X.append(features)
                y.append(label_idx)

    # 2. 转换格式 (OpenCV 必须用 float32)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    print(f"样本总数: {len(X)}")

    # 划分验证集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ==========================================
    # 3. 创建 SVM 并使用【自动训练】(关键修改!)
    # ==========================================
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_RBF)
    # 注意：这里不再手动设置 setC 和 setGamma，交给 trainAuto 去找

    print("正在进行自动调参训练 (trainAuto)...")

    # === 核心修改：使用 trainAuto 代替 train ===
    # kFold=5 表示它会把数据分成5份交叉验证，确保参数最稳
    svm.trainAuto(X_train, cv2.ml.ROW_SAMPLE, y_train, kFold=5)

    # 4. 验证准确率
    _, y_pred = svm.predict(X_test)
    matches = (y_pred.flatten().astype(int) == y_test)
    acc = np.count_nonzero(matches) / len(y_test)

    print(f"----------------------------------")
    print(f"OpenCV SVM 模型最终准确率: {acc * 100:.2f}%")
    print(f"----------------------------------")

    # 5. 保存
    if not os.path.exists('models'):
        os.makedirs('models')

    xml_path = 'models/svm_model.xml'
    svm.save(xml_path)
    print(f"SVM 模型已保存至: {xml_path}")


if __name__ == '__main__':
    train_svm_xml()
