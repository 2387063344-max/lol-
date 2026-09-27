"""入口脚本：把 data/raw/docs 下的攻略文档切片并写入 Chroma 向量库。

用法：
    python build_kb.py            # 构建 / 追加
    python build_kb.py --rebuild  # 全量重建（换嵌入模型后必须执行）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.lol_agent.ingest.build_index import main  # noqa: E402

if __name__ == "__main__":
    main()
