import os
from PIL import Image

def get_image_resolutions(folder_path):
    resolutions = []
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            file_path = os.path.join(folder_path, file_name)
            with Image.open(file_path) as img:
                resolutions.append((file_name, img.size))
    return resolutions

folder_path = '/mnt/Xnf/DataSets/CLIC/professional_test_2020'
image_resolutions = get_image_resolutions(folder_path)

large_pix_num = 0
large_img_info = {}

for image_name, resolution in image_resolutions:
    pix_num = resolution[0] * resolution[1]
    print(f"{image_name}: {resolution[0]}x{resolution[1]}")
    
    if pix_num > large_pix_num:
        large_pix_num = pix_num
        large_img_info = {
            "img_name": image_name,
            "resolution": resolution
        }

print(large_img_info)