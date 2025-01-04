from setuptools import setup, find_packages

setup(
    name="llm_decisions",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'pydantic>=2.0.0',
        'PyYAML>=5.1',
        'aiofiles>=0.6.0',
        'ollama>=0.1.0',
    ],
    python_requires='>=3.8',
    author="Your Name",
    author_email="your.email@example.com",
    description="A package for evaluating language model decisions",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)