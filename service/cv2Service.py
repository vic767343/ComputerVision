import os
import cv2
import numpy as np
from datetime import datetime

# CV2Service: Computer Vision Service using OpenCV

class CV2Service:
    def __init__(self):
        self.firstProcessimage_dir = 'statics/images/firstProcessimage/'
        self.secProcessimage_dir = 'statics/images/secProcessimage/'
        # 確保目標資料夾存在
        os.makedirs(self.secProcessimage_dir, exist_ok=True)

    def get_last_counter(self, date_str):
        """獲取當天最後使用的編號"""
        max_counter = 0
        try:
            for filename in os.listdir(self.secProcessimage_dir):
                # 檢查文件名格式：編號(3位)_日期(8位)+擴展名
                if len(filename) >= 12 and filename[:3].isdigit() and filename[4:12] == date_str:
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
        
        for filename in os.listdir(self.firstProcessimage_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                filepath = os.path.join(self.firstProcessimage_dir, filename)
                try:
                    # 使用 OpenCV 讀取圖片
                    img = cv2.imread(filepath)
                    if img is None:
                        print(f"無法讀取圖片: {filename}")
                        continue
                    
                    # 影像尺寸調整：調整到 640x640
                    img = cv2.resize(img, (640, 640))
                    
                    # Letterbox 處理：將圖片放入固定大小的框中，填充邊緣
                    h, w = img.shape[:2]
                    target_size = 640
                    if h != w:
                        diff = abs(h - w)
                        pad1 = diff // 2
                        pad2 = diff - pad1
                        if h > w:
                            img = cv2.copyMakeBorder(img, 0, 0, pad1, pad2, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                        else:
                            img = cv2.copyMakeBorder(img, pad1, pad2, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                    
                    # 尺寸標準化：確保為 640x640
                    img = cv2.resize(img, (target_size, target_size))
                    
                    # 灰階轉換
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
                    # 模糊化：高斯模糊
                    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                    
                    # 影像增強：直方圖均衡化
                    enhanced = cv2.equalizeHist(blurred)
                    
                    # 生成新文件名：三位碼編號_ + 日期(YYYYMMDD) + 原始檔案副檔名
                    name, ext = os.path.splitext(filename)
                    new_filename = f"{counter:03d}_{date_str}{ext}"
                    
                    # 儲存處理後的圖片
                    output_path = os.path.join(self.secProcessimage_dir, new_filename)
                    cv2.imwrite(output_path, enhanced)
                    print(f"處理完成: {filename} -> {new_filename}")
                    
                    # 編號遞增
                    counter += 1
                    
                except Exception as e:
                    print(f"處理 {filename} 時發生錯誤: {str(e)}")

# 使用範例
if __name__ == "__main__":
    service = CV2Service()
    service.process_images()
