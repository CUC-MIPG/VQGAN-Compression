import os

test_cmds = [
    {"config": "results/test_config/JpegAI_VQ64_serial_edge1_down.yaml", "name": "JpegAI_VQ64"},
    {"config": "results/test_config/JpegAI_VQ64_serial_edge2_down.yaml", "name": "JpegAI_VQ64"},
    {"config": "results/test_config/JpegAI_VQ64_serial_nomask.yaml", "name": "JpegAI_VQ64"},
]

gpu = "7,"

for cfg in test_cmds:
    cmd = f"python test.py --base {cfg['config']} --gpus {gpu} --name {cfg['name']}"
    os.system(cmd)