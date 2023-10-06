from setuptools import setup, find_packages

setup(
    name="synthlung",
    packages=find_packages(),
    version='0.1.0',
    author="Vemund Fredriksen and Svein Ole Matheson Sevle",
    url="https://github.com/VemundFredriksen/SynthLung",
    license="Apache License",
    python_requires='>=3.7',
    install_requires=[
        'numpy>=1.23.5',
        'monai>=1.1.0',
        'torch>=2.0.1',
    ],
    entry_points={
        'console_scripts': [
            'synthlung = synthlung.__main__:main'
        ]
    },
    classifiers=[
         "Programming Language :: Python :: 3.7",
         "Programming Language :: Python :: 3.8",
         "Programming Language :: Python :: 3.9",
         "Programming Language :: Python :: 3.10",
         "License :: OSI Approved :: Apache License",
         "Operating System :: OS Independent",
     ],
)