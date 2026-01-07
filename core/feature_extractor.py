# 文件路径: core/feature_extractor.py
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops


def extract_features(image_path=None, image_data=None):
    """
    题目标准要求的 12 维特征提取：
    [0-2]: RGB均值 (3维)
    [3-5]: HSV均值 (3维)
    [6-11]: GLCM纹理 (6维: 对比度, 相关性, 能量, 同质性, 熵, 差异性)
    """
    # 1. 读取图像逻辑
    if image_data is not None:
        img = image_data
    elif image_path is not None:
        img = cv2.imread(image_path)
    else:
        return None

    if img is None: return None

    # 2. 统一尺寸 (固定尺度对纹理计算很重要)
    img = cv2.resize(img, (224, 224))

    # ==========================
    # Part 1: 颜色特征 (6维)
    # ==========================
    # RGB 均值
    # OpenCV 读进来是 BGR，注意顺序
    b_mean, g_mean, r_mean = cv2.mean(img)[:3]

    # HSV 均值
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_img)
    h_mean = np.mean(h)
    s_mean = np.mean(s)
    v_mean = np.mean(v)

    # ==========================
    # Part 2: 纹理特征 (6维)
    # ==========================
    # 转为灰度图用于计算 GLCM
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 计算 GLCM (距离=1, 角度=0度-水平, 灰度级=256)
    # normed=True 是必须的，否则熵和能量的计算数量级不对
    glcm = graycomatrix(gray, [1], [0], levels=256, symmetric=True, normed=True)

    # 1. 对比度 (Contrast)
    contrast = graycoprops(glcm, 'contrast')[0, 0]

    # 2. 相关性 (Correlation)
    correlation = graycoprops(glcm, 'correlation')[0, 0]

    # 3. 能量 (Energy / ASM)
    energy = graycoprops(glcm, 'energy')[0, 0]

    # 4. 同质性 (Homogeneity)
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

    # 5. 差异性 (Dissimilarity)
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]

    # 6. 熵 (Entropy) - skimage 没有直接函数，需手动计算
    # 公式: -sum(p * log2(p))
    P = glcm[:, :, 0, 0]  # 获取归一化矩阵
    # 加上一个极小值 1e-10 防止 log(0) 报错
    entropy = -np.sum(P * np.log2(P + 1e-10))

    # ==========================
    # Part 3: 组合输出 (12维)
    # ==========================
    feature_vector = [
        r_mean, g_mean, b_mean,  # 1-3
        h_mean, s_mean, v_mean,  # 4-6
        contrast, correlation, energy, homogeneity, entropy, dissimilarity  # 7-12
    ]

    return feature_vector