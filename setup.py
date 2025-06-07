import io

from setuptools import find_packages
from setuptools import setup


setup(
    name="biassearch",
    version="1.0.0",
    url="http://127.0.0.1:5000",
    # license="MIT",
    maintainer="Olav Larsen Halleraker & Guillermo Rodrigo Pérez",
    # maintainer_email="ohallera@gmail.com guillerodper@gmail.com",
    description="Flask app for Retrieval and Classification in the course Web Information Retrieval at Tsinghua University",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=["flask"],
)