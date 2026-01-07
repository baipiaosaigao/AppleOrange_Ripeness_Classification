import os


def rename_files():
    # 数据集根目录
    root_dir = "./dataset/test"

    if not os.path.exists(root_dir):
        print(f"错误：找不到目录 {root_dir}")
        return

    # 遍历下面的 6 个子文件夹
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)

        # 确保是文件夹
        if not os.path.isdir(folder_path):
            continue

        print(f"正在处理文件夹: {folder_name} ...")

        files = os.listdir(folder_path)
        count = 0

        for i, filename in enumerate(files):
            # 获取旧文件的完整路径
            old_path = os.path.join(folder_path, filename)

            # 获取文件后缀 (比如 .jpg, .png)
            file_ext = os.path.splitext(filename)[1].lower()

            # 如果没有后缀，或者不是图片，跳过
            if file_ext not in ['.jpg', '.jpeg', '.png', '.bmp']:
                continue

            # 构建新名字: 文件夹名_序号.后缀
            # 例如: apple_ripe_0.jpg
            new_filename = f"{folder_name}_{i}{file_ext}"
            new_path = os.path.join(folder_path, new_filename)

            # 重命名
            try:
                os.rename(old_path, new_path)
                count += 1
            except Exception as e:
                print(f"  无法重命名 {filename}: {e}")

        print(f"  -> 完成！重命名了 {count} 张图片")


if __name__ == "__main__":
    rename_files()