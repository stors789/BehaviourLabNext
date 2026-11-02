"""python -m behaviourlab 入口。"""

import multiprocessing
import sys

from .launcher import main

if __name__ == "__main__":
    # 冻结 multiprocessing 自举先于 main：避免误入 GUI。
    multiprocessing.freeze_support()
    sys.exit(main())
