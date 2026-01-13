import cv2
import numpy as np
# 必须引入 skimage 来计算真正的 GLCM
from skimage.feature import graycomatrix, graycoprops


# def extract_features(image_path=None, image_data=None):
#     """
#     提取图像的 12 维特征 (符合 RGB均值 + HSV统计 + GLCM对比度/相关性/能量):
#     1. 颜色特征 (9维): RGB均值(3) + HSV均值(3) + HSV标准差(3)
#     2. 纹理特征 (3维): Contrast, Correlation, Energy
#     """
#     # ==============================
#     # 0. 图像读取与预处理
#     # ==============================
#     if image_data is not None:
#         img = image_data
#     elif image_path is not None:
#         try:
#             # 处理中文路径读取
#             img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
#         except Exception:
#             return None
#     else:
#         return None
#
#     if img is None:
#         return None
#
#     # 统一大小
#     img = cv2.resize(img, (224, 224))
#
#     # ==============================
#     # 1. 颜色特征 (RGB + HSV) -> 共9维
#     # ==============================
#     # [1-3] RGB 均值
#     mean_rgb = np.mean(img, axis=(0, 1))  # [B, G, R]
#
#     # [4-6] HSV 均值
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     mean_hsv = np.mean(hsv, axis=(0, 1))  # [H, S, V]
#
#     # [7-9] HSV 标准差 (对应“分量占比”的统计特性)
#     std_hsv = np.std(hsv, axis=(0, 1))  # [H_std, S_std, V_std]
#
#     # ==============================
#     # 2. 纹理特征 (GLCM) -> 共3维
#     # ==============================
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     # 灰度级量化 (压缩到16级，提高速度)
#     n_levels = 16
#     gray_quantized = (gray // (256 // n_levels)).astype(np.uint8)
#
#     # 计算 GLCM 矩阵
#     glcm = graycomatrix(gray_quantized, distances=[1], angles=[0], levels=n_levels, symmetric=True, normed=True)
#
#     # 提取 3 个核心纹理统计量 (按您的要求：对比度、相关性、能量)
#
#     # [10] Contrast (对比度)
#     glcm_contrast = graycoprops(glcm, 'contrast')[0, 0]
#
#     # [11] Correlation (相关性) - 替换了之前的 Homogeneity
#     # 范围通常是 -1 到 1
#     glcm_correlation = graycoprops(glcm, 'correlation')[0, 0]
#
#     # [12] Energy (能量)
#     glcm_energy = graycoprops(glcm, 'energy')[0, 0]
#
#     # ==============================
#     # 3. 特征融合与归一化 (输出 12 维)
#     # ==============================
#     feats = [
#         # --- 颜色特征 (归一化到 0-1) ---
#         mean_rgb[2] / 255.0, mean_rgb[1] / 255.0, mean_rgb[0] / 255.0,  # R, G, B
#         mean_hsv[0] / 180.0, mean_hsv[1] / 255.0, mean_hsv[2] / 255.0,  # H, S, V
#         std_hsv[0] / 50.0, std_hsv[1] / 50.0, std_hsv[2] / 50.0,  # HSV Std
#
#         # --- 纹理特征 (GLCM) ---
#         # Contrast (0-225) -> 缩放
#         glcm_contrast / 50.0,
#
#         # Correlation (-1到1) -> 归一化到 0-1 之间 ((val+1)/2) 或者直接用
#         # 为了SVM稳定性，建议简单平移缩放: (x + 1) / 2
#         (glcm_correlation + 1.0) / 2.0,
#
#         # Energy (0-1) -> 直接用
#         glcm_energy
#     ]
#
#     return feats


import cv2
import numpy as np
# 必须引入 skimage 来计算 GLCM
from skimage.feature import graycomatrix, graycoprops


def extract_features(image_path=None, image_data=None):
    """
    提取严格的 12 维特征 (6颜色 + 6纹理):
    Color:   R_mean, G_mean, B_mean, H_mean, S_mean, V_mean
    Texture: Contrast, Correlation, Energy, Entropy, Homogeneity, Dissimilarity
    """
    # ==============================
    # 0. 图像读取与预处理
    # ==============================
    if image_data is not None:
        img = image_data
    elif image_path is not None:
        try:
            # 处理中文路径读取
            img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
        except Exception:
            return None
    else:
        return None

    if img is None:
        return None

    # 统一大小 (建议 224x224，既保留纹理细节又不会太慢)
    img = cv2.resize(img, (224, 224))

    # ==============================
    # 1. 颜色特征 (6维)
    # ==============================

    # --- RGB 均值 (3维) ---
    # OpenCV 默认读取为 BGR，需调整为 R, G, B 顺序
    mean_bgr = np.mean(img, axis=(0, 1))
    rgb_mean = [mean_bgr[2], mean_bgr[1], mean_bgr[0]]

    # --- HSV 均值 (3维) ---
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_mean = np.mean(hsv, axis=(0, 1))  # [H, S, V]

    # ==============================
    # 2. 纹理特征 (6维) - GLCM
    # ==============================
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 灰度级量化 (压缩到 16 级，提高速度和鲁棒性)
    n_levels = 16
    gray_quantized = (gray // (256 // n_levels)).astype(np.uint8)

    # 计算 GLCM
    # distances=[1]: 步长为1
    # angles=[0]: 水平方向 0度
    # levels=16: 灰度级
    # symmetric=True, normed=True: 对称且归一化(概率矩阵)
    glcm = graycomatrix(gray_quantized, distances=[1], angles=[0], levels=n_levels, symmetric=True, normed=True)

    # --- 提取 skimage 自带的 5 个特征 ---
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]

    # --- 手动计算 GLCM 熵 (Entropy) ---
    # 公式: -sum(p * log2(p))
    # 避免 log(0) 的错误，只计算非零元素
    p = glcm
    p_nonzero = p[p > 0]
    entropy = -np.sum(p_nonzero * np.log2(p_nonzero))

    # ==============================
    # 3. 特征融合与归一化 (12维)
    # ==============================
    feats = [
        # --- 颜色特征 (6个) ---
        rgb_mean[0] / 255.0,  # R Mean
        rgb_mean[1] / 255.0,  # G Mean
        rgb_mean[2] / 255.0,  # B Mean

        hsv_mean[0] / 180.0,  # H Mean (0-180)
        hsv_mean[1] / 255.0,  # S Mean
        hsv_mean[2] / 255.0,  # V Mean

        # --- 纹理特征 (6个) ---
        # Contrast (0 ~ (levels-1)^2) -> 缩放
        contrast / 50.0,

        # Correlation (-1 ~ 1) -> 映射到 0~1
        (correlation + 1.0) / 2.0,

        # Energy (0 ~ 1) -> 直接用
        energy,

        # Entropy (通常 0 ~ log2(levels)*2) -> 简单缩放
        entropy / 5.0,

        # Homogeneity (0 ~ 1) -> 直接用
        homogeneity,

        # Dissimilarity (0 ~ levels-1) -> 缩放
        dissimilarity / 10.0
    ]

    return feats
