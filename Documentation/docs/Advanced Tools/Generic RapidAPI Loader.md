---
sidebar_position: 5
--- 

# Generic RapidAPI Loader

This tool helps calling any API on RapidAPI using a generic function, offering methods for both synchronous and asynchronous requests. This makes it easier for developers who need to integrate multiple API services into their applications. This client manages API keys, request headers, and error handling internally, reducing the overhead for the developers and allowing them to focus on implementing the core functionality.

To use it, you'll need to input your API key in the `.env` file in this form:

```
RAPIDAPI_API_KEY="your_api_key"
```

## Synchronous API Calls (`call_api`)

The `call_api` method allows users to perform synchronous HTTP requests to any specified RapidAPI endpoint. This method is useful when the application requires a direct response from the API for further processing.

### Parameters
- `api_url`: String specifying the full URL of the API endpoint.
- `method`: HTTP method to use (e.g., 'GET', 'POST'), default is 'GET'.
- `params` (Optional): Dictionary of query parameters for the 'GET' request.
- `headers_extra` (Optional): Dictionary to provide additional headers.
- `data` (Optional): Dictionary specifying data for 'POST' requests.
- `json` (Optional): Dictionary specifying JSON payload for 'POST' requests.
- `max_retries` (Optional): Maximum number of retries
- `backoff_factor` (Optional): Factor by which the delay increases during each retry

The response from the API is checked for its HTTP status code. If the response indicates success (e.g., 200, 201), the method returns the JSON data. If the response is not successful, it raises an HTTP error with the status code and message.

**This function includes automatic retries** for client connection errors (e.g., timeouts, server not available). The method will attempt to resend the request for a default number of 3 times with a backoff factor of 2. 

If you wnat to change these values you can include them in the parameters when sending an API call. 

### Sample Use Case

Let's try using the [Domain Authority API](https://rapidapi.com/hassan.cs91/api/domain-authority1/playground/apiendpoint_f2c2bcde-e9c2-45aa-9d0c-47d6b21b876b) which is an API i developed that returns the domain power, organic clicks, average rank, and keywords rank for any domain name.

```python
from  SimplerLLM.tools.rapid_api import RapidAPIClient

api_url = "https://domain-authority1.p.rapidapi.com/seo/get-domain-info"
api_params = {
    'domain': 'learnwithhasan.com',
}

api_client = RapidAPIClient() 
response = api_client.call_api(api_url, method='GET', params=api_params)
```

## Asynchronous API Calls (`call_api_async`)

The `call_api_async` function is designed for making asynchronous API calls. This is particularly useful in environments that support asynchronous operations, allowing non-blocking API calls that can greatly improve the efficiency of your application.

### Parameters
- `api_url`: String specifying the full URL of the API endpoint.
- `method`: HTTP method to use (e.g., 'GET', 'POST'), default is 'GET'.
- `params` (Optional): Dictionary of query parameters for the 'GET' request.
- `headers_extra` (Optional): Dictionary to provide additional headers.
- `data` (Optional): Dictionary specifying data for 'POST' requests.
- `json` (Optional): Dictionary specifying JSON payload for 'POST' requests.
- `max_retries` (Optional): Maximum number of retries
- `backoff_factor` (Optional): Factor by which the delay increases during each retry

As you can see it takes the same parameters as the `call_api`, but each request is handled asynchronously.

The response from the API is checked for its HTTP status code. If the response indicates success (e.g., 200, 201), the method returns the JSON data. If the response is not successful, it raises an HTTP error with the status code and message.

**This function includes automatic retries** for client connection errors (e.g., timeouts, server not available). The method will attempt to resend the request for a default number of 3 times with a backoff factor of 2. 

If you want to change these values you can include them in the parameters when sending an API call. 

### Sample Use Case

Let's try using the [Domain Authority API](https://rapidapi.com/hassan.cs91/api/domain-authority1/playground/apiendpoint_f2c2bcde-e9c2-45aa-9d0c-47d6b21b876b) which is an API i developed that returns the domain power, organic clicks, average rank, and keywords rank for any domain name.

```python
from  SimplerLLM.tools.rapid_api import RapidAPIClient

api_url = "https://domain-authority1.p.rapidapi.com/seo/get-domain-info"
api_params = {
    'domain': 'learnwithhasan.com',
}

api_client = RapidAPIClient() 
response = api_client.call_api_async(api_url, method='GET', params=api_params)
```

That's how you can benefit from SimplerLLM to make RapidAPI calling Simpler!