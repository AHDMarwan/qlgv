from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="qlgv",
    version="0.1.0",
    author="AHDMarwan",
    author_email="your.email@example.com",
    description="Quantum Local Gradient Variance Toolkit for analyzing trainability of Variational Quantum Circuits",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AHDMarwan/qlgv",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pennylane>=0.28.0",
        "torch>=1.9.0",
        "numpy>=1.19.0",
        "pandas>=1.1.0",
    ],
    keywords="quantum computing variational circuits gradient variance trainability",
    project_urls={
        "Bug Reports": "https://github.com/AHDMarwan/qlgv/issues",
        "Source": "https://github.com/AHDMarwan/qlgv",
    },
)
