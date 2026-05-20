from setuptools import setup, find_packages

setup(
    name="glomeralign",
    version="0.1.0",
    description="Align and compare in vivo and ex vivo brain images with glomeruli identification",
    packages=find_packages(),
    package_data={
        "glomeralign": ["config/*.yaml"],
    },
    python_requires=">=3.9",
    install_requires=[
        "napari==0.5.5",
        "PyQt5",
        "PyYAML",
        "numpy",
        "scipy",
        "scikit-image",
        "opencv-python",
        "pandas",
        "tifffile",
        "cellpose",
    ],
    entry_points={
        "console_scripts": [
            "glomeralign=glomeralign.gui:main",
        ],
    },
)
