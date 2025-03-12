from setuptools import setup, find_packages

setup(
    name='ninetails',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'h5py',
        'pyyaml',
        'imageio'
    ],
    extras_require={
        'gpu': ['cupy'],  # Users can install with GPU support using `pip install ninetails[gpu]`
    },
    author='Antoine C.D. Hoffmann',
    author_email='ahoffman@pppl.gov',
    description='A 9 gyromoment fluid code',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/python_utilities',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)