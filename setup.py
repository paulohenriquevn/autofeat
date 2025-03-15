from setuptools import setup, find_packages
import os
import site
import sys


# Adicione o diretório do usuário ao sys.path
site_user_dirs = site.getusersitepackages()
if site_user_dirs not in sys.path:
    sys.path.append(site_user_dirs)

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cafe-autofe",
    version="0.1.3",
    author="CAFE Team",
    author_email="your.email@example.com",
    description="Component Automated Feature Engineer - Sistema de Engenharia Automática de Features",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/cafe-autofe",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "joblib>=1.1.0",
        "networkx>=3.4.2",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "black>=21.5b2",
            "isort>=5.9.1",
            "flake8>=3.9.2",
        ],
    },
)