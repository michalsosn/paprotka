from setuptools import setup, find_packages


setup(
    name="paprotka",
    version="0.1",
    description='Various machine learning utilities.',
    author='Michał Sośnicki',
    author_email='sosnicki.michal@gmail.com',
    license='MIT',
    packages=find_packages(),
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
)
