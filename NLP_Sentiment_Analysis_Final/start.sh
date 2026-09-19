#!/bin/bash
# 中文文本情感分析系统 - Mac/Linux 一键启动脚本

# 设置终端编码
export LANG="zh_CN.UTF-8"
export LC_ALL="zh_CN.UTF-8"

echo "================================================================================"
echo "                 中文文本情感分析系统 - Mac/Linux 一键启动"
echo "================================================================================"
echo ""

# 检查Python是否安装
if ! command -v python3 &> /dev/null; then
    echo "[错误] 未检测到 Python3 环境！"
    echo ""
    echo "请先安装 Python 3.8 或更高版本："
    echo "  Mac: brew install python3"
    echo "  Ubuntu/Debian: sudo apt install python3 python3-pip"
    echo "  CentOS/RHEL: sudo yum install python3 python3-pip"
    echo ""
    exit 1
fi

echo "[信息] 检测到 Python 环境"
python3 --version
echo ""

# 切换到脚本所在目录
cd "$(dirname "$0")"

# 检查虚拟环境是否存在
if [ ! -f "venv/bin/python" ]; then
    echo "[信息] 正在创建虚拟环境..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "[警告] 创建虚拟环境失败，将使用系统Python"
        PYTHON_CMD="python3"
    else
        echo "[信息] 虚拟环境创建成功"
        PYTHON_CMD="venv/bin/python"
    fi
else
    echo "[信息] 检测到虚拟环境"
    PYTHON_CMD="venv/bin/python"
fi

echo ""

# 检查依赖是否已安装
$PYTHON_CMD -c "import flask, sklearn, pandas, jieba" &> /dev/null
if [ $? -ne 0 ]; then
    echo "[信息] 正在安装依赖包（首次运行较慢，请耐心等待）..."
    echo "[信息] 使用清华镜像源加速下载..."
    echo ""
    
    $PYTHON_CMD -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
    $PYTHON_CMD -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    
    if [ $? -ne 0 ]; then
        echo ""
        echo "[错误] 依赖安装失败！"
        echo "请尝试手动运行：pip install -r requirements.txt"
        echo ""
        exit 1
    fi
    
    echo ""
    echo "[信息] 依赖安装完成！"
else
    echo "[信息] 依赖已安装，跳过安装步骤"
fi

echo ""

# 初始化目录
echo "[信息] 初始化项目目录..."
$PYTHON_CMD config.py &> /dev/null
echo "[信息] 目录初始化完成"

echo ""

# 检查模型文件是否存在
if [ ! -f "models/svm_model.pkl" ]; then
    echo "[警告] 未检测到模型文件，正在训练模型..."
    echo "[信息] 这可能需要几分钟时间..."
    echo ""
    
    $PYTHON_CMD train/train_model.py
    
    if [ $? -ne 0 ]; then
        echo ""
        echo "[错误] 模型训练失败！"
        echo ""
        exit 1
    fi
    
    echo ""
    echo "[信息] 模型训练完成！"
else
    echo "[信息] 检测到模型文件"
fi

echo ""
echo "================================================================================"
echo "[信息] 正在启动 Web 服务..."
echo "[信息] 服务地址：http://localhost:5000"
echo "[信息] 按 Ctrl+C 可停止服务"
echo "================================================================================"
echo ""

# 自动打开浏览器（延迟2秒）
sleep 2
if command -v open &> /dev/null; then
    # Mac
    open http://localhost:5000
elif command -v xdg-open &> /dev/null; then
    # Linux
    xdg-open http://localhost:5000 &
fi

# 启动Flask服务
$PYTHON_CMD app.py

echo ""
echo "[信息] 服务已停止"
