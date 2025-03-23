#!/usr/bin/env python
"""
MR Thermometry 应用程序启动脚本
该脚本用于从项目根目录启动应用程序
"""
import os
import sys

# 将项目根目录添加到Python路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# 从mvc.main导入main函数
from mvc.main import main

if __name__ == '__main__':
    main()