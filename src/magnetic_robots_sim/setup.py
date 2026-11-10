import os
from glob import glob

from setuptools import setup

package_name = 'magnetic_robots_sim'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='vladimir',
    maintainer_email='user@example.com',
    description='Closed-loop simulation of magnetic robot manipulation',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'continuum_sim_node = magnetic_robots_sim.continuum_sim_node:main',
        ],
    },
)