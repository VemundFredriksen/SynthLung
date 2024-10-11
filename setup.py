import setuptools

setuptools.setup(
    name="synthlung",
    version="0.1.0",
    author="Vemund Fredriksen and Svein Ole Matheson Sevle",
    author_email="vemund.fredriksen@hotmailcom",
    description="Package for generating synthetic lung tumors",
    url="https://github.com/VemundFredriksen/SynthLung",
    packages=setuptools.find_packages(),
    entry_points={"console_scripts": ["synthlung = synthlung.__main__:main"]},
    install_requires=[
        "numpy==1.26.2",
        "torch==2.4.1",
        "tqdm==4.66.5",
        "monai==1.3.2",
        "lungmask==0.2.20",
        "pytest==8.3.3",
        "pytest-cov==5.0.0",
        "requests==2.32.3",
        "nibabel==5.3.0",
        "scikit-image==0.24.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)