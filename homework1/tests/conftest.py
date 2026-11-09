"""pytest 路径引导：把 src/ 加入 sys.path，使测试可直接 import 各模块。"""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
