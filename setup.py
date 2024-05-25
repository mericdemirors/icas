from setuptools import setup, find_packages

with open("USAGE.md", "r") as f:
    usage_desc = f.read()

setup(
    name="label-img",
    version="1.0",
    packages=find_packages(),
    description="Tool for labeling images",
    install_requires=[
        "numpy>=1.26.4",
        "opencv_contrib_python>=4.9.0.80", # be careful about other opencv libraries on your device
        "scikit-image>=0.22.0",
        "tqdm>=4.66.2",
        "matplotlib>=3.8.4",
        "mplcursors>=0.5.3",
        "datasketch>=1.6.4",
        "Pillow>=10.3.0",
        "ImageHash>=4.3.1",
        # "torch>=2.3.0",
        # "scikit-learn>=1.4.2",
        # "torchvision>=0.18.0",
        # "segment-anything>=1.0"
    ],
    long_description=usage_desc,
    long_description_content_type="text/markdown",
    author="mericdemirors",
    author_email="demirorsmeric@gmail.com",
    url="https://github.com/mericdemirors/labelimg")
