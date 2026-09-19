@echo off
chcp 65001 >nul
title 中文文本情感分析系统

echo ================================================================================
echo                  中文文本情感分析系统 - Windows 一键启动
echo ================================================================================
echo.

:: 检查Python是否安装
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未检测到 Python 环境！
    echo.
    echo 请先安装 Python 3.8 或更高版本：
    echo 下载地址：https://www.python.org/downloads/
    echo.
    echo 安装时请勾选 "Add Python to PATH"
    echo.
    pause
    exit /b 1
)

echo [信息] 检测到 Python 环境
python --version
echo.

:: 切换到脚本所在目录
cd /d "%~dp0"

:: 检查虚拟环境是否存在
if not exist "venv\Scripts\python.exe" (
    echo [信息] 正在创建虚拟环境...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [警告] 创建虚拟环境失败，将使用系统Python
        set PYTHON_CMD=python
    ) else (
        echo [信息] 虚拟环境创建成功
        set PYTHON_CMD=venv\Scripts\python
    )
) else (
    echo [信息] 检测到虚拟环境
    set PYTHON_CMD=venv\Scripts\python
)

echo.

:: 检查依赖是否已安装
%PYTHON_CMD% -c "import flask, sklearn, pandas, jieba" >nul 2>&1
if %errorlevel% neq 0 (
    echo [信息] 正在安装依赖包（首次运行较慢，请耐心等待）...
    echo [信息] 使用清华镜像源加速下载...
    echo.
    
    %PYTHON_CMD% -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
    %PYTHON_CMD% -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    
    if %errorlevel% neq 0 (
        echo.
        echo [错误] 依赖安装失败！
        echo 请尝试手动运行：pip install -r requirements.txt
        echo.
        pause
        exit /b 1
    )
    
    echo.
    echo [信息] 依赖安装完成！
) else (
    echo [信息] 依赖已安装，跳过安装步骤
)

echo.

:: 初始化目录
echo [信息] 初始化项目目录...
%PYTHON_CMD% config.py >nul 2>&1
echo [信息] 目录初始化完成

echo.

:: 检查模型文件是否存在
if not exist "models\svm_model.pkl" (
    echo [警告] 未检测到模型文件，正在训练模型...
    echo [信息] 这可能需要几分钟时间...
    echo.
    
    %PYTHON_CMD% train\train_model.py
    
    if %errorlevel% neq 0 (
        echo.
        echo [错误] 模型训练失败！
        echo.
        pause
        exit /b 1
    )
    
    echo.
    echo [信息] 模型训练完成！
) else (
    echo [信息] 检测到模型文件
)

echo.
echo ================================================================================
echo [信息] 正在启动 Web 服务...
echo [信息] 服务地址：http://localhost:5000
echo [信息] 按 Ctrl+C 可停止服务
echo ================================================================================
echo.

:: 自动打开浏览器（延迟2秒）
start "" http://localhost:5000

:: 启动Flask服务
%PYTHON_CMD% app.py

echo.
echo [信息] 服务已停止
pause
