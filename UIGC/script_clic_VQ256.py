import os

test_cmds = [
    {"config": "results/test_config/CLIC_VQ256_serial_edge1_down.yaml", "name": "CLIC_VQ256"},
    {"config": "results/test_config/CLIC_VQ256_serial_edge2_down.yaml", "name": "CLIC_VQ256"},
    {"config": "results/test_config/CLIC_VQ256_serial_nomask.yaml", "name": "CLIC_VQ256"},
]

gpu = "2,"

for cfg in test_cmds:
    cmd = f"python test.py --base {cfg['config']} --gpus {gpu} --name {cfg['name']}"
    ret = os.system(cmd)
    assert ret == 0, "Fail!"