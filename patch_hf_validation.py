#!/usr/bin/env python3
"""
补丁脚本：禁用HuggingFace Hub的路径验证
解决本地模型路径无法通过validate_repo_id验证的问题
"""

import sys
import os

# 设置环境变量
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

# Monkey patch huggingface_hub的validate_repo_id函数
try:
    from huggingface_hub.utils import _validators

    # 保存原始函数
    _original_validate_repo_id = _validators.validate_repo_id

    def patched_validate_repo_id(repo_id: str, *args, **kwargs):
        """
        修补后的validate_repo_id：允许本地路径
        """
        # 如果是绝对路径，直接返回
        if os.path.isabs(repo_id):
            return
        # 否则调用原始验证
        return _original_validate_repo_id(repo_id, *args, **kwargs)

    # 替换函数
    _validators.validate_repo_id = patched_validate_repo_id
    print("[Patch] Successfully patched huggingface_hub.utils._validators.validate_repo_id")

except Exception as e:
    print(f"[Patch] Warning: Could not patch validate_repo_id: {e}")
    print("[Patch] Continuing anyway...")

# 运行主训练脚本
if __name__ == "__main__":
    # 导入verl训练器
    from verl.trainer.main_ppo import main

    # 运行训练
    main()
