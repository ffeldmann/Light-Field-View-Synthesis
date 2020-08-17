from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="LightFieldViewSynthesis",
    version="0.0.1",
    author="Felix Feldmann",
    author_email="felix@bnbit.de",
    description="Light Field View Synthesis using VAE",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ffeldmann/Light-Field-View-Synthesis",
    include_package_data=True,
    packages=find_packages(),
    install_requires=["numpy", "tqdm", "matplotlib", "Pillow", "pytorch"],
    classifiers=["Programming Language :: Python :: 3", ],
)
