import io
import os
from setuptools import setup, find_packages


setup(
    name="snfa",
    version="0.0.1",
    description="A simple neural forced aligner for phoneme to audio alignment",
    author="Patchethium",
    author_email="asguftuikh@gmail.com",
    url="https://github.com/Patchethium/snfa",
    packages=find_packages("snfa"),
    install_requires=["numpy"]
)