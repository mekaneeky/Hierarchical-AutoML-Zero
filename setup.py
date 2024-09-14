from setuptools import setup, find_packages

setup(
    name="automl_zero",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "tqdm",
        "numpy",
        "bittensor",
        "huggingface_hub",
        "python-dotenv",
    ],
    extras_require={
        "dev": ["pytest", "black", "flake8"],
    },
    entry_points={
        "console_scripts": [
            "automl_mine=automl.main_activation_2:main",
            "automl_validate=automl.validator:main",
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="An AutoML-Zero implementation for evolving neural network activation functions",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/automl_zero",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)