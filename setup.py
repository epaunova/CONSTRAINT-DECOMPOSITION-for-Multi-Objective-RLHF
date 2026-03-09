"""Setup script for constraint-decomposition package."""

from pathlib import Path
from setuptools import setup, find_packages

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    requirements = [
        line.strip() 
        for line in requirements_path.read_text().split("\n")
        if line.strip() and not line.startswith("#")
    ]
else:
    requirements = [
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "trl>=0.7.0",
        "accelerate>=0.24.0",
    ]

setup(
    name="constraint-decomposition",
    version="0.1.0",
    author="Eva Paunova",
    author_email="e.hpaunova@gmail.com",
    description="Constraint Decomposition for Multi-Objective RLHF in Large Language Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/epaunova/constraint-decomposition",
    project_urls={
        "Bug Tracker": "https://github.com/epaunova/constraint-decomposition/issues",
        "Documentation": "https://github.com/epaunova/constraint-decomposition#readme",
    },
    license="Apache-2.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.9.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords=[
        "rlhf",
        "llm",
        "reinforcement-learning",
        "multi-objective",
        "alignment",
        "constraint-decomposition",
    ],
)
