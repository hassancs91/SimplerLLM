from setuptools import setup, find_packages


# Read requirements (excluding comments and empty lines)
with open("requirements.txt") as f:
    requirements = [
        line.strip() for line in f.read().splitlines()
        if line.strip() and not line.strip().startswith('#')
    ]

# Optional dependencies for specific features
extras_require = {
    "transformers": [
        "transformers>=4.36.0",
        "torch>=2.0.0",
        "accelerate>=0.25.0",
    ],
}

# Read the long description from the README file
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="SimplerLLM",
    version="0.3.3.5",
    author="Hasan Aboul Hasan",
    author_email="hasan@learnwithhasan.com",
    description="An easy-to-use Library for interacting with language models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hassancs91/SimplerLLM",
    packages=find_packages(),
    install_requires=requirements,
    extras_require=extras_require,
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
