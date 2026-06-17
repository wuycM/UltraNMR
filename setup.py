import os
from setuptools import find_namespace_packages, setup


here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


setup(
    name="ultranmr",
    version="0.1.0",
    description="UltraNMR models and utilities for NMR representation learning and downstream tasks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_namespace_packages(
        include=[
            "models*",
            "utils*",
            "layers*",
            "losses*",
            "functional_group_prediction*",
        ]
    ),
    include_package_data=True,
    install_requires=[
        "torch==2.2.1",
        "numpy==1.25.0",
        "tqdm==4.68.1",
        "lmdb==2.2.1",
        "pyyaml==6.0.3",
        "rdkit==2023.9.6",
        "transformers==4.46.3",
        "tokenizers==0.20.3",
        "tensorboard==2.20.0",
        "scikit-learn==1.5.2",
        "pytorch-lightning==2.0.8",
        "torchmetrics==1.3.2",
        "lightning-utilities==0.15.3",
    ],
    python_requires=">=3.11,<3.12",
)
