import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyspacer",  # Replace with your own username
    version="0.2.9",
    author="Oscar Beijbom",
    author_email="oscar.beijbom@gmail.com",
    description="Spatial image analysis with caffe and pytorch backends.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/beijbom/pyspacer",
    packages=setuptools.find_packages(exclude=['scripts',
                                               'scripts.*',
                                               'scripts.aws.*',
                                               'scripts.regression.*',
                                               'spacer.tests',
                                               'spacer.tests.*']),
    classifiers=[
        "Programming Language :: Python :: 3.5",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
    install_requires=[
        'wget==3.2',
        'tqdm==4.43.0',
        'Pillow>=4.2.0',
        'numpy>=1.17.5',
        'scikit-learn==0.22.1',
        'scikit-image==0.15.0',
        'torch==1.4.0',
        'torchvision==0.5.0',
        'boto3',
        'botocore'
    ]
)
