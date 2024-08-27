---
sidebar_position: 2
--- 

# Search Engine Integration

This section provides an overview of how SimplerLLM facilitates the integration of Google and DuckDuckGo search engines in your code. It includes functions that allow applications to retrieve search results directly through APIs provided by Google and the free-to-use DuckDuckGo API.

The Data is returned in form of a `SearchResult` object which is designed to standardize the format of search results across different search engines. This object includes the following fields:
- `URL` : The URL of the search result.
- `Domain`: The domain name extracted from the URL.
- `Title`: The title of the search result.
- `Description`: A brief description associated with the result.

## Google Search Integration

Google search integration is supported through two paid APIs: the Serper API and the Value SERP API. These APIs provide excellent search capabilities and are suitable for applications requiring good and stable search functionalities.

### Serper API Functions

To use the Serper API functions, you'll need to have your Serper API Key in the `.env` file in your project folder in this format:

```
SERPER_API_KEY="your_serper_api_key"
```

**Synchronous `search_with_serper_api` Function**

It Takes 2 parameters:
- `query` (string): The search query.
- `num_results` (int, Optional): The maximum number of results to return, default is 50.

It returns a list of `SearchResult` objects depending on the maximum number of results you specifiy. Here's an example usage:

```python
from SimplerLLM.tools.serp import search_with_serper_api

search_results = search_with_serper_api("What is SEO", 5)

print(search_results)
```

**Asynchronous `search_with_serper_api_async` Function**

It's the same as the normal function, however it Asynchronously fetches search results from Google using the Serper API.

It also takes the same 2 parameters, and returns a list of `SearchResult` objects depending on the maximum number of results you specifiy. Here's an example usage:

```python
import asyncio
from SimplerLLM.tools.serp import search_with_serper_api_async

async def fetch_results():
    search_result = await search_with_serper_api_async("Latest AI advancements", 5)
    print(search_result)

asyncio.run(fetch_results())
```

### Value SERP API Functions

To use the Value SERP API functions, you'll need to have your Value SERP API Key in the `.env` file in your project folder in this format:

```
VALUE_SERP_API_KEY="your_value_serp_api_key"
```

**Synchronous `search_with_value_serp_api` Function**

It takes 2 parameters:
- `query` (string): The search query.
- `num_results` (int, Optional): The maximum number of results to return, default is 50.

It returns a list of `SearchResult` objects depending on the maximum number of results you specify. Here's an example usage:

```python
from SimplerLLM.tools.serp import search_with_value_serp_api

search_results = search_with_value_serp_api("What is SEO", 5)

print(search_results)
```

**Asynchronous `search_with_value_serp_api_async` Function**

It's the same as the normal function, however it asynchronously fetches search results from Google using the Value SERP API.

It also takes the same 2 parameters, and returns a list of `SearchResult` objects depending on the maximum number of results you specify. Here's an example usage:

```python
import asyncio
from SimplerLLM.tools.serp import search_with_value_serp_api_async

async def fetch_results():
    search_results = await search_with_value_serp_api_async("Latest AI advancements", 5)
    print(search_results)

asyncio.run(fetch_results())
```

## DuckDuckGo API Functions

Unlike Google Search which is integrated using paid APIs, the DuckDuckGo search integration avaiable through their own free to use API. However, DuckDuckGo prioritizes user privacy and doesn’t track search history, leading to less personalized search results. This can make its results less relevant compared to Google, which customizes searches using a lot of user data.

**Synchronous `search_with_duck_duck_go` Function**

It takes 2 parameters:
- `query` (string): The search query.
- `max_results` (int, Optional): The maximum number of results to return, default is 10.

It returns a list of `SearchResult` objects depending on the maximum number of results you specify. Here's an example usage:

```python
from SimplerLLM.tools.serp import search_with_duck_duck_go

search_results = search_with_duck_duck_go("Open source projects", 10)

print(search_results)
```

**Asynchronous `search_with_duck_duck_go_async` Function**

It's the same as the normal function but fetches search results from DuckDuckGo asynchronously.

It also takes the same 2 parameters, and returns a list of `SearchResult` objects depending on the maximum number of results you specify. Here's an example usage:

```python
import asyncio
from SimplerLLM.tools.serp import search_with_duck_duck_go_async

async def fetch_results():
    search_results = await search_with_duck_duck_go_async("Open source tools", 10)
    print(search_results)

asyncio.run(fetch_results())
```

That's how you can benefit from SimplerLLM to make Search Engine Integration Simpler!