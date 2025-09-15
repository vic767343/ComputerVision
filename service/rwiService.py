import os
import exifread
from PIL import Image
import albumentations as A
import numpy as np
from datetime import datetime

# RWIService: Read, Write, Image Service

class RWIService:
    def __init__(self):
        self.origin_dir = 'statics/images/originImage/'
        self.processed_dir = 'statics/images/firstProcessimage/'
        # 確保目標資料夾存在
        os.makedirs(self.processed_dir, exist_ok=True)

        # 一次處理的照片命名法則：三位碼編號 + 日期(YYYYMMDD) + 原始檔案副檔名
    def get_last_counter(self, date_str):
        """獲取當天最後使用的編號"""
        max_counter = 0
        try:
            for filename in os.listdir(self.processed_dir):
                # 檢查文件名格式：編號(3位) + 日期(8位) + 擴展名
                if len(filename) >= 11 and filename[:3].isdigit() and filename[3:11] == date_str:
                    counter = int(filename[:3])
                    if counter > max_counter:
                        max_counter = counter
        except FileNotFoundError:
            pass
        return max_counter

    def process_images(self):
        # 獲取當前日期
        date_str = datetime.now().strftime("%Y%m%d")
        
        # 獲取當天最後使用的編號
        last_counter = self.get_last_counter(date_str)
        counter = last_counter + 1
        
        for filename in os.listdir(self.origin_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                filepath = os.path.join(self.origin_dir, filename)
                try:
                    # 讀取 EXIF 並修正
                    with open(filepath, 'rb') as f:
                        tags = exifread.process_file(f)
                    
                    # 這裡可以添加具體的 EXIF 修正邏輯，例如移除 GPS 信息或調整方向
                    # 例如：如果有方向標籤，可以旋轉圖片
                    
                    # 使用 Pillow 讀取圖片
                    img = Image.open(filepath)
                    
                    # 色彩轉換：確保為 RGB 模式
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # 使用 Albumentations 做資料增強
                    transform = A.Compose([
                        A.Rotate(limit=45, p=0.5),  # 隨機旋轉
                        A.HorizontalFlip(p=0.5),    # 水平翻轉
                        A.VerticalFlip(p=0.2),      # 垂直翻轉
                        A.RandomBrightnessContrast(p=0.3),  # 隨機亮度和對比度
                        A.GaussianBlur(blur_limit=3, p=0.2),  # 高斯模糊
                    ])
                    
                    # 將 PIL 圖片轉換為 numpy array
                    img_array = np.array(img)
                    
                    # 應用增強
                    augmented = transform(image=img_array)
                    img_aug = Image.fromarray(augmented['image'])
                    
                    # 生成新文件名：三位碼編號 + 日期
                    name, ext = os.path.splitext(filename)
                    new_filename = f"{counter:03d}{date_str}{ext}"
                    
                    # 儲存處理後的圖片
                    output_path = os.path.join(self.processed_dir, new_filename)
                    img_aug.save(output_path)
                    print(f"處理完成: {filename} -> {new_filename}")
                    
                    # 編號遞增
                    counter += 1
                    
                except Exception as e:
                    print(f"處理 {filename} 時發生錯誤: {str(e)}")

# 使用範例
if __name__ == "__main__":
    service = RWIService()
    service.process_images()
