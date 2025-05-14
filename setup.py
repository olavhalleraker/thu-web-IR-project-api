import io

from setuptools import find_packages
from setuptools import setup

# with io.open("README.rst", "rt", encoding="utf8") as f:
#     readme = f.read()

setup(
    name="biassearch",
    version="1.0.0",
    url="http://127.0.0.1:5000",
    # license="MIT",
    maintainer="Olav Larsen Halleraker",
    # maintainer_email="ohallera@gmail.com",
    description="Flask app for Retrieval and Classification in the course Web Information Retrieval at Tsinghua University",
    # long_description=readme,
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=["flask"],
    # extras_require={"test": ["pytest", "coverage"]},
)