import os
from glob import glob
from setuptools import setup, find_packages

package_name = 'current_predictor'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),        glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'),        glob('config/*.yaml')),
        (os.path.join('share', package_name, 'model_weights'), glob('model_weights/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Michał Błotniak',
    maintainer_email='michal.blotniak@onet.pl',
    description='80 kW power limiter current predictor (ARX Q=0.90 + XGBoost Q=0.90)',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'current_predictor_node = current_predictor.node:main',
        ],
    },
)
