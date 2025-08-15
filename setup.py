from setuptools import setup, find_packages
from src.interpret_lightgbm.__init__ import __version__

setup(
    name='interpret_lightgbm',
    version=__version__,
    author='Your Name',
    author_email='mohamedmxo7@gmail.com',
    description='Interpret-lightgbm turns powerful LightGBM models into transparent, explainable predictors.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/m-elgebaly/interpret-lightgbm',
    packages=find_packages(where='src'),  # Look for packages inside the 'src' folder
    package_dir={'': 'src'},             # Map the package root to the 'src' directory
    install_requires=[
        'numpy>=1.23,<2.3',
        'pandas>=1.4',
        'scipy>=1.10',
        'scikit-learn>=1.7',
        'lightgbm>=4.0',
        'shap>=0.47',
        'matplotlib>=3.5',
        #'tqdm>=4.65',
        #'ipython>=8.0'
    ],
    python_requires='>=3.6',
)
