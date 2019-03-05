from setuptools import setup

setup(
    name='santander',
    version='0.0.1',
    entry_points={
        'console_scripts': [
            'santander=main:run'
        ]
    }
)