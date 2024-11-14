from setuptools import find_packages, setup

setup(
    name="abreast",
    version="1.0",
    description="AXTI Breast Shape Template generator",
    author="Marta Pinto, Koen Michielsen",
    packages=find_packages("abreast"),
    install_requires=[
        "numpy>=2.1.3",
        "scipy>=1.14",
        "tifffile>=2024.9.20",
    ],
    include_package_data=True,
    python_requires=">=3.10",
    license_files=("LICENSE", "abreast/data/LICENSE")
)