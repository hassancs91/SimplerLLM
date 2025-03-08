from setuptools import setup, find_packages


# Read requirements
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# Read the long description from the README file
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="SimplerLLM",
    version="0.3.1.13",
    author="Hasan Aboul Hasan",
    author_email="hasan@learnwithhasan.com",
    description="An easy-to-use Library for interacting with language models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hassancs91/SimplerLLM",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.6",
    license='MIT',
    keywords="text generation, openai, LLM, RAG",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],


    # Add additional fields as necessary
)
