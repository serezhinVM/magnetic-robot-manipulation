from setuptools import setup

package_name = 'continuum_rod_model'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'numpy', 'scipy'],
    zip_safe=True,
    maintainer='vladimir',
    maintainer_email='user@example.com',
    description='Quasi-static Cosserat rod model for magnetic soft robots',
    license='MIT',
    tests_require=['pytest'],
    scripts=['scripts/plot_rod.py'],
)
