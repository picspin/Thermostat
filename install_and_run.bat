@echo off
echo MR热测量系统 - 安装和运行脚本
echo ==============================
echo.


REM Check if Python is installed
where python >nul 2>&1
if %errorlevel% equ 0 (
    echo Using 'python' command...
    
    REM Create virtual environment if it doesn't exist
    if not exist "venv" (
        echo Creating virtual environment...
        python -m venv venv
    )
    
    REM Activate virtual environment
    call venv\Scripts\activate.bat
    
    REM Upgrade pip
    python -m pip install --upgrade pip
    
    REM Install dependencies
    python -m pip install -r requirements.txt
    
    echo.
    echo ===================================================================
    echo Starting MR Thermometry application...
    echo ===================================================================
    echo.
    python run.py
) else (
    echo Python not found! Please install Python 3.7 or later.
    pause
    exit /b 1
)

pause

@REM :: 检查Python是否已安装
@REM python --version >nul 2>&1
@REM if %errorlevel% neq 0 (
@REM     echo 错误: 未找到Python。请安装Python 3.7或更高版本。
@REM     pause
@REM     exit /b 1
@REM )

@REM :: 检查是否已存在虚拟环境
@REM if not exist venv\ (
@REM     echo 创建虚拟环境...
@REM     python -m venv venv
@REM     if %errorlevel% neq 0 (
@REM         echo 错误: 无法创建虚拟环境。
@REM         pause
@REM         exit /b 1
@REM     )
@REM )

@REM :: 激活虚拟环境并安装依赖
@REM echo 激活虚拟环境并安装依赖...
@REM call venv\Scripts\activate.bat
@REM python -m pip install --upgrade pip
@REM pip install -r requirements.txt
@REM if %errorlevel% neq 0 (
@REM     echo 错误: 安装依赖失败。
@REM     pause
@REM     exit /b 1
@REM )

@REM :: 运行应用程序
@REM echo 运行MR热测量系统...
@REM python run.py

@REM :: 退出虚拟环境
@REM call venv\Scripts\deactivate.bat
@REM pause 