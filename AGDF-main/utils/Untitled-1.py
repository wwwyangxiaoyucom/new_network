
import os
import PIL.Image as Image
def is_image_valid(file_path):
    try:
        img = Image.open(file_path)
        img.verify()  # 检查图片文件的完整性
        return True
    except (IOError, SyntaxError):
        return False
def filter_and_delete_invalid_images(dataset_path):
    invalid_images = []
    
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(root, file)
            if not is_image_valid(file_path):
                invalid_images.append(file_path)
                os.remove(file_path)  # 删除无法打开的图片文件
    
    return invalid_images
dataset_path = '/home/yangyufang/AGPCNet-main/data/MDFA/training'
invalid_images = filter_and_delete_invalid_images(dataset_path)

print(f"删除了 {len(invalid_images)} 张无法打开的图片.")
