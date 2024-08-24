---
sidebar_position: 1
slug: /
---

# Introduction

SimplerLLM is an open-source Python library designed to simplify interactions with Large Language Models (LLMs). 

Whether you’re a developer working on AI-powered applications or a beginner getting started with ways to benefit from LLMs, SimplerLLM provides a unified interface and powerful tools that make it super easy to integrate AI into your projects. 

Having functionalities that simplify complex tasks like data extraction, text chunking, and vector database integration, SimplerLLM allows you to focus on building optimal AI workflows and projects rather than getting lost in configurations.

### Examples of Projects Made Easy with SimplerLLM:

- [Semantic Chunking Guide](https://learnwithhasan.com/what-is-semantic-chunking/)
- [How to Build a Plagiarism Detector](https://learnwithhasan.com/how-to-build-a-semantic-plagiarism-detector/)
- [Find Similar Research Paper Abstracts](https://learnwithhasan.com/find-similar-research-paper-abstracts/)
- [Get Consistent Structured JSON from LLMs](https://learnwithhasan.com/consistent-json-gemini-python/)
- [Build an AI Paraphraser Tool](https://learnwithhasan.com/ai-paraphraser-tool/)
- [Topic Research with AI](https://learnwithhasan.com/topic-research-with-ai/)
- [Repurpose Content with a single click](https://learnwithhasan.com/repurpose-content-ai-python/)
- [Create a SEO Tool for Automated Blog Interlinking](https://learnwithhasan.com/ai-seo-tool-interlinking/)
- [And Much More!](https://learnwithhasan.com/blog/)

## Main Functionalies

SimplerLLM is designed to make coding literally **Simpler**:smile: Here are some of the key features it provides:
- **Unified Interface**: Manage interactions with multiple Large Language Models (OpenAI, Google Gemini, Ollamma Local Model, etc...) from one place.
- **Search Engine Integrations**: Integrate search engine functionalities from Google and DuckDuckGo directly into your applications.
- **Generic Loader**: Load text from web content like articles, YouTube video transcripts, and traditional formats like PDF, CSV, and DOCX with one click. In addition to Writing to files with simple functions.
- **Vector Embeddings & Databases**: Easily create vector embeddings and add them into a vector database.
- **Multiple Chunking Functions**: Advanced text processing to split texts into chunks by size, sentences, or even semantics.
- **Dynamic Prompt Templates**: Use prompt templates with dynamic inputs and easily replace placeholder values with desired content.
- **Image Generation**: Create high-quality images tailored to your needs using the Stability API.
- **Image Storage**: Easily save images from the web directly to your device.
- **YouTube Data Extraction**: Extract metadata and detailed transcripts from any YouTube video.
- **AI Agent Interaction & Creation**: Simplify setting up and customizing AI agents for personalized applications.

## Setup SimplerLLM

To get started with SimplerLLM, follow these steps:

#### **1. Install the Library**

First, install SimplerLLM using `pip`:

```bash
pip install simplerllm
```

#### **2. Set Up Your Environment (.env File)**

To use this library, you need to set several API keys in your environment. Start by creating a `.env` file in the root directory of your project and adding your API keys there.

🔴 **Important:** This file should be kept private and not committed to version control (e.g., add it to your `.gitignore` file) to protect your keys.

Here is an example of what your `.env` file would look like:

```
OPENAI_API_KEY="your_openai_api_key_here" # For accessing OpenAI's API
GEMINI_API_KEY="your_gemeni_api_key_here" # For accessing Gemini's API
ANTHROPIC_API_KEY="your_claude_api_key_here" # For accessing Anthropic's API
RAPIDAPI_API_KEY="your_rapidapi_api_key_here" # For accessing APIs on RapidAPI
VALUE_SERP_API_KEY="your_value_serp_api_key_here" # For Google search
SERPER_API_KEY="your_serper_api_key_here" # For Google search
LWH_API_KEY="your_lwh_api_key" # For LLM Playground on LearnWithHasan
LWH_USER_ID="your_lwh_user_id" # For LLM Playground on LearnWithHasan
STABILITY_API_KEY="your_stability_api_key_here" # For image generation
```

Each of these keys corresponds to a specific functionality within SimplerLLM. Depending on which features you plan to use, you can add only the relevant keys.

## Contribute and Collaborate

If you need help using SimplerLLM or have questions about building projects with it, there are several ways to get support and connect with the community:

### 1. Join Our Discord Community

We have an active Discord server where users can:
- Ask questions about the library and receive help from other users.
- Share projects and ideas for using SimplerLLM in creative ways.
- Report bugs and get assistance with troubleshooting issues.

Whether you’re a beginner or an experienced developer, the Discord community is a great place to learn and collaborate.

[Join the SimplerLLM Discord Server](https://discord.com/invite/HUrtZXyp3j)

### 2. Contribute to the Project

SimplerLLM is an open-source project, and we welcome contributions from the community! If you notice an issue in the documentation or want to improve the codebase, you can easily contribute:

- On any documentation page, click the **"Edit this page"** link at the bottom.
- This will take you to the GitHub repository, where you can fork the project and make your changes.
- Once your changes are ready, submit a pull request for review.
- If the changes are approved, they will be merged and reflected on the site.

Your contributions help improve SimplerLLM and make it more useful for everyone. We value every suggestion, fix, and improvement made by the community.
