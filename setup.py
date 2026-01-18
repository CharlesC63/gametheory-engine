"""
GameTheory Engine Package Setup

Install with: pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name="gametheory-engine",
    version="0.1.0",
    description="Unified game theory framework for Chess, Shogi, Poker, and Go with finance applications",
    author="Charles",
    author_email="",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
        ],
        "viz": [
            "matplotlib>=3.4",
            "plotly>=5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "kuhn-cfr=src.games.poker.kuhn_cfr:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Games/Entertainment :: Board Games",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
