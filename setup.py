from setuptools import setup

setup(
    name='plot_hgbdt',
    version='0.1.0',
    py_modules=['plot_hgbdt'],
    install_requires=[
        'scikit-learn',
        'lightgbm',
        'graphviz',
    ],
)
