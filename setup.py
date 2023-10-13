import setuptools

setuptools.setup(
    name="synthlung",
    version="0.0.1",
    author="Vemund Fredriksen and Svein Ole Matheson Sevle",
    author_email="vemund.fredriksen@hotmailcom",
    description="Package for generating synthetic lung tumors",
    url="https://github.com/VemundFredriksen/SynthLung",
    packages=setuptools.find_packages(),
    entry_points={"console_scripts": ["synthlung = synthlung.__main__:main"]},
    install_requires=[
        "numpy",
        "torch",
        "tqdm",
        "monai"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)