from setuptools import setup, find_packages

setup(
    name="mr_thermometry",
    version="1.0.0",
    description="MR热测量系统 - 用于磁共振成像热图分析的应用程序",
    author="MR Thermometry Team",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
        "PyQt5>=5.15.0",
        "pydicom>=2.3.0",
        "SimpleITK>=2.1.0",
        "scikit-image>=0.18.0",
        "scipy>=1.6.0",
    ],
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "mr_thermometry=mvc.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
) 