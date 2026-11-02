"""根目录冻结入口：与 ``python -m behaviourlab`` 委托同一 main。

两种源码入口与打包 EXE 最终进入同一个主入口，私有 worker 分派、
数据根所有权与清理链只有一份实现。
"""

import multiprocessing
import sys

from behaviourlab.launcher import main

if __name__ == "__main__":
    # 冻结 multiprocessing 自举先于 main：避免误入 GUI。
    multiprocessing.freeze_support()
    sys.exit(main())
