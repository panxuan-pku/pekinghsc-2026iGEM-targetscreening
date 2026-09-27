"""M14: 训练 GEARS 因果扰动模型（Dixit Perturb-seq 数据集）
自有工作；GEARS 本身来自 snap-stanford（登记见 ATTRIBUTIONS.md）
"""
import time
from gears import PertData, GEARS
from paths import GEARS_DATA_DIR, OUTPUT_DIR
from pathlib import Path

print("1/4 下载并加载 Dixit Perturb-seq 数据（首次会下载）...")
pert_data = PertData(GEARS_DATA_DIR)
pert_data.load(data_name="dixit")

print("2/4 数据划分（simulation split）...")
pert_data.prepare_split(split="simulation", seed=1)
pert_data.get_dataloader(batch_size=32, test_batch_size=128)

print("3/4 初始化 GEARS 模型（CPU, hidden_size=64）...")
gears_model = GEARS(pert_data, device="cpu")
gears_model.model_initialize(hidden_size=64)

print("4/4 训练（15 epochs）...")
t0 = time.time()
gears_model.train(epochs=15)
print(f"训练耗时 {(time.time()-t0)/60:.1f} 分钟")

target = Path(OUTPUT_DIR) / "gears_dixit_v1"
target.parent.mkdir(parents=True, exist_ok=True)
gears_model.save_model(str(target))
print(f"✅ 模型已保存: {target}")
