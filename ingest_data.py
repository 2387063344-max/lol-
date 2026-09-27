"""入口脚本：把官方 Data Dragon 数据采进 SQLite。

用法：
    python ingest_data.py                 # 采集最新版本（含背景故事）
    python ingest_data.py --version 16.19.1
    python ingest_data.py --no-lore       # 跳过背景故事，更快
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.lol_agent.ingest.ddragon import main  # noqa: E402

if __name__ == "__main__":
    main()
