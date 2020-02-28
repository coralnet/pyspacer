import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyspacer-BEIJBOM",  # Replace with your own username
    version="1.0.0",
    author="Oscar Beijbom",
    author_email="oscar.beijbom@gmail.com",
    description="Spatial image analysis with caffe and pytorch backends.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/beijbom/pyspacer",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.5",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='==3.5.*',
)