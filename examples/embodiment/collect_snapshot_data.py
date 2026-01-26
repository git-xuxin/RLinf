# collect_data/examples/embodiment/collect_snapshot_data.py

import os
import time
import datetime
import cv2
import hydra
import numpy as np
import torch
from tqdm import tqdm

from rlinf.envs.realworld.realworld_env import RealWorldEnv
from rlinf.envs.realworld.common.keyboard.keyboard_listener import KeyboardListener
from rlinf.scheduler import Cluster, ComponentPlacement, Worker

class SnapshotCollector(Worker):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        self.env = RealWorldEnv(
            cfg.env.eval,
            num_envs=1,
            seed_offset=0,
            total_num_processes=1,
            worker_info=self.worker_info,
        )
        
        self.listener = KeyboardListener()
        self.log_path = self.cfg.runner.logger.log_path
        
        # --- 初始化按键映射和文件夹 ---
        self.key_mapping = {} # 格式: {'key_char': 'folder_name'}
        
        snapshot_cfg = self.cfg.runner.get("snapshot_config", None)
        mode = snapshot_cfg.get("mode", "default") if snapshot_cfg else "default"
        
        if mode == "custom" and snapshot_cfg and "mappings" in snapshot_cfg:
            # 使用 YAML 中定义的自定义映射
            self.log_info("Initializing Custom Snapshot Mode...")
            for key, folder_name in snapshot_cfg.mappings.items():
                self.key_mapping[str(key)] = folder_name
        else:
            # 默认模式 (兼容旧代码)
            self.log_info("Initializing Default Snapshot Mode (s=success, f=failure)...")
            self.key_mapping = {
                's': 'success',
                'f': 'failure'
            }

        # 创建所有需要的文件夹
        for folder_name in self.key_mapping.values():
            full_path = os.path.join(self.log_path, folder_name)
            os.makedirs(full_path, exist_ok=True)
            self.log_info(f"Initialized folder: {full_path}")
            
        self.log_info(f"Snapshot Collector Initialized.")
        self.log_info(f"Key Mappings: {self.key_mapping}")

    def save_frame(self, obs, folder_name):
        """
        保存帧到指定文件夹名称
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        save_dir = os.path.join(self.log_path, folder_name)
        
        # 提取 main_images
        if "main_images" in obs:
            img_tensor = obs["main_images"][0] 
            img_np = img_tensor.cpu().numpy().astype(np.uint8)
            
            # --- 修复颜色问题 ---
            # 环境输出是 RGB，OpenCV 保存需要 BGR
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            filename = f"{timestamp}.png"
            filepath = os.path.join(save_dir, filename)
            
            success = cv2.imwrite(filepath, img_bgr)
            if success:
                self.log_info(f"Saved snapshot to [{folder_name}]: {filename}")
            else:
                self.log_info(f"Failed to save snapshot to {filepath}")

        # 处理额外视角 (如果有)
        if "extra_view_images" in obs:
            extra_imgs = obs["extra_view_images"][0]
            for i in range(extra_imgs.shape[0]):
                img_np = extra_imgs[i].cpu().numpy().astype(np.uint8)
                # --- 修复颜色问题 ---
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                
                filename = f"{timestamp}_view_{i}.png"
                filepath = os.path.join(save_dir, filename)
                cv2.imwrite(filepath, img_bgr)

    def run(self):
        obs, _ = self.env.reset()
        action_dim = 6 if self.cfg.env.eval.get("no_gripper", True) else 7
        
        self.log_info("Start collecting snapshots...")
        self.log_info("Auto-reset is DISABLED. Move freely.")
        self.log_info(f"Active keys: {list(self.key_mapping.keys())}")
        
        try:
            while True:
                action = np.zeros((1, action_dim), dtype=np.float32)
                
                # 步进环境
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                
                # 获取按键输入
                key = self.listener.get_key()
                
                # --- 动态检查按键 ---
                if key in self.key_mapping:
                    target_folder = self.key_mapping[key]
                    self.save_frame(obs, target_folder)
                    time.sleep(0.2) # 防止按一次键保存多张
                
                # --- 修复 Reset 问题 ---
                # 无论 terminated 或 truncated 状态如何，永远不调用 reset()
                # 始终延续当前的观测
                obs = next_obs
                    
        except KeyboardInterrupt:
            self.log_info("Stopping collector...")
        finally:
            self.env.close()

@hydra.main(
    version_base="1.1", config_path="config", config_name="collect_data_snapshot"
)
def main(cfg):
    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = ComponentPlacement(cfg, cluster)
    env_placement = component_placement.get_strategy("env")
    
    collector = SnapshotCollector.create_group(cfg).launch(
        cluster, name=cfg.env.group_name, placement_strategy=env_placement
    )
    collector.run().wait()

if __name__ == "__main__":
    main()