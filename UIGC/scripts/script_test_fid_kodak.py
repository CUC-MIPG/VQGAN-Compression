import os
import pandas as pd
from test_fid import calculate_fid

job_list = [
    {
        "path": "results/tf_result/Kodak_VQ16_edge_down_mask_2023-11-16T14-54-53", 
        "name": "Kodak_VQ16_edge"
    },
    {
        "path": "results/tf_result/Kodak_VQ16_edge_more_down_mask_2023-11-16T15-11-46", 
        "name": "Kodak_VQ16_edge_more"
    },
    {
        "path": "results/tf_result/Kodak_VQ16_entire_down_mask_2023-11-16T14-52-02", 
        "name": "Kodak_VQ16_nomask"
    },
    {
        "path": "results/tf_result/Kodak_VQ64_edge_down_mask_2023-11-20T13-45-25", 
        "name": "Kodak_VQ64_edge"
    },
    {
        "path":"results/tf_result/Kodak_VQ64_edge_more_down_mask_2023-11-20T13-46-01", 
        "name": "Kodak_VQ64_edge_more"
    },
    {
        "path": "results/tf_result/Kodak_VQ64_entire_down_mask_2023-11-20T13-46-50", 
        "name": "Kodak_VQ64_nomask"
    },
    {
        "path": "results/tf_result/Kodak_VQ256_edge_down_mask_2023-11-20T14-04-41", 
        "name": "Kodak_VQ256_edge"
    },
    {
        "path":"results/tf_result/Kodak_VQ256_edge_more_down_mask_2023-11-20T14-05-09", 
        "name": "Kodak_VQ256_edge_more"
    },
    {
        "path": "results/tf_result/Kodak_VQ256_entire_down_mask_2023-11-20T14-05-35", 
        "name": "Kodak_VQ256_nomask"
    },
]

reference_dir = "/mnt/Xnf/DataSets/Kodak"
img_dir_name = "image"
excel_dir_name= "excel"
excel_pfx = "average"
pfx = "predicted"
device = "cuda:7"

pd_data = pd.DataFrame()

for idx, item in enumerate(job_list):
    dir = item['path']
    name = item['name']
    img_dir = os.path.join(dir, img_dir_name)
    excel_dir = os.path.join(dir, excel_dir_name)
    
    excel_file_list = os.listdir(excel_dir)
    excel_avg_file = None
    for i in excel_file_list:
        if i.startswith(excel_pfx):
            excel_avg_file = os.path.join(excel_dir, i)
    
    avg_data = pd.read_excel(excel_avg_file, index_col=0)
    fid = calculate_fid(reference_dir, img_dir, pfx, device)
    bpp = avg_data.loc['total_compressed_bpp'].item()
    
    res_dict = {
        "name": name,
        "bpp": bpp,
        "fid": fid
    }
    
    this_data = pd.DataFrame(res_dict, index=[idx])
    pd_data = pd.concat([pd_data, this_data])
    
pd_data.to_excel("Kodak_fid.xlsx")