#!/bin/bash

echo "MR热测量系统 - 安装和运行脚本"
echo "=============================="

# 检查Python是否已安装
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python。请安装Python 3.7或更高版本。"
    exit 1
fi

# 检查是否已存在虚拟环境
if [ ! -d "venv" ]; then
    echo "创建虚拟环境..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "错误: 无法创建虚拟环境。"
        exit 1
    fi
fi

# 激活虚拟环境并安装依赖
echo "激活虚拟环境并安装依赖..."
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "错误: 安装依赖失败。"
    exit 1
fi

# 运行应用程序
echo "运行MR热测量系统..."
python run.py

# 退出虚拟环境
deactivate 