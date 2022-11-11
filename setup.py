from setuptools import setup, find_packages

setup(
    name='nanograd',
    author='Asif Ahmed',
    description='Re-implementation of micrograd library with pytorch api like autograd engine and neural nets.',
    version='0.1.0',
    url='https://github.com/quickgrid/nanograd',
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
    install_requires=[
        'rich'
    ]
)
