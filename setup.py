from setuptools import setup, find_packages

setup(
    name="admet-toxicity-predictor",
    version="1.0.0",
    description="ADMET toxicity prediction with two-layer chemistry intuition validation",
    author="Your Name",
    python_requires=">=3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "rdkit>=2023.3.1",
        "deepchem>=2.7.1",
        "torch>=2.0.0",
        "shap>=0.42.0",
        "xgboost>=1.7.0",
        "matplotlib>=3.7.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "pytest-cov>=4.1.0", "ruff>=0.0.280", "mypy>=1.4.0"],
        "notebooks": ["jupyter>=1.0.0", "ipywidgets>=8.0.0", "plotly>=5.14.0"],
    },
)
