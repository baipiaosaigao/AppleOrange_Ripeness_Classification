import cv2
import numpy as np


def extract_features(image_path=None, image_data=None):
    """
    提取图像的颜色和纹理特征 (经过归一化处理)
    """
    if image_data is not None:
        img = image_data
    elif image_path is not None:
        # 处理中文路径读取
        try:
            img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
        except Exception:
            return None
    else:
        return None

    if img is None:
        return None

    # 统一大小，减少计算量
    img = cv2.resize(img, (224, 224))

    # ==============================
    # 1. 颜色特征 (RGB + HSV)
    # ==============================
    # 计算 RGB 均值
    mean_rgb = np.mean(img, axis=(0, 1))  # [B, G, R]

    # 转为 HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mean_hsv = np.mean(hsv, axis=(0, 1))  # [H, S, V]
    std_hsv = np.std(hsv, axis=(0, 1))  # [H_std, S_std, V_std]

    # ==============================
    # 2. 纹理特征 (GLCM - 灰度共生矩阵)
    # ==============================
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # GLCM 比较慢，我们只算中心区域或者缩小后算
    # 这里简单起见，对整个灰度图计算几个统计量代替复杂的 GLCM
    # (真正的 GLCM 在 Python 里慢，且 OpenCV 没有原生实现，这里用统计特征模拟纹理)

    # 对比度 (Contrast): 也就是标准差
    texture_contrast = np.std(gray)

    # 熵 (Entropy): 图像信息的复杂程度
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist / hist.sum()  # 归一化直方图
    # 避免 log(0)
    hist = hist[hist > 0]
    texture_entropy = -np.sum(hist * np.log2(hist))

    # 平滑度/能量 (Energy)
    texture_energy = np.sum(hist ** 2)

    # ==============================
    # 3. 特征融合与【归一化】
    # ==============================
    # 颜色是 0-255，我们需要除以 255 让它变成 0-1
    feats = [
        mean_rgb[2] / 255.0, mean_rgb[1] / 255.0, mean_rgb[0] / 255.0,  # R, G, B (0-1)
        mean_hsv[0] / 180.0, mean_hsv[1] / 255.0, mean_hsv[2] / 255.0,  # H, S, V (H是0-180)
        std_hsv[0] / 50.0, std_hsv[1] / 50.0, std_hsv[2] / 50.0,  # Std 稍微缩放一下

        # 纹理特征通常数值不大，或者需要单独缩放
        texture_contrast / 100.0,  # 对比度通常在 0-100 左右
        texture_entropy / 10.0,  # 熵通常在 0-8 左右
        texture_energy  # 能量本来就是 0-1
    ]

    # 替换掉原来的 12 维特征
    return feats
