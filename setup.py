"""Setup configuration for NBA prediction system."""
from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="nba-prediction-system",
    version="4.0.0",
    description="NBA sports betting prediction system using machine learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Green Bier Ventures",
    python_requires=">=3.11",
    packages=find_packages(where="."),
    package_dir={"": "."},
    install_requires=[
        "httpx==0.27.2",
        "requests==2.32.3",
        "python-dotenv==1.0.1",
        "pyyaml==6.0.2",
        "pandas==2.2.3",
        "numpy==1.26.4",
        "pydantic==2.9.2",
        "tenacity==9.0.0",
        "scikit-learn==1.5.2",
        "joblib==1.4.2",
        "rapidfuzz==3.10.0",
    ],
    extras_require={
        "dev": [
            "pytest==8.3.3",
            "black",
            "flake8",
            "mypy",
        ],
        "api": [
            "fastapi==0.115.0",
            "uvicorn[standard]==0.31.0",
        ],
        "advanced_ml": [
            "xgboost==2.1.1",
            "lightgbm==4.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "nba-predict=scripts.predict:main",
            "nba-train=scripts.train_models:main",
            "nba-backtest=scripts.backtest:main",
            "nba-collect-odds=scripts.run_the_odds_tomorrow:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
