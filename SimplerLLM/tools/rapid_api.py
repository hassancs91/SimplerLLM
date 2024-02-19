from dotenv import load_dotenv
import os
import time
import requests
import aiohttp
import asyncio
from typing import Optional, Any, Dict

load_dotenv()  # Load the environment variables

class RapidAPIClient:
    def __init__(self, api_key: Optional[str] = None, timeout: int = 30):
        """
        Initialize the RapidAPI client.

        :param api_key: Optional API key. If not provided, it will be read from the environment variable 'RAPID_API_KEY'.
        :param timeout: Request timeout in seconds.
        """
        self.api_key = api_key if api_key else os.getenv('RAPIDAPI_API_KEY')
        self.timeout = timeout

        if not self.api_key:
            raise ValueError("API key must be provided or set as an environment variable 'RAPID_API_KEY'")

    def _construct_headers(self, api_url: str, headers_extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """
        Construct headers for the API call.

        :param api_url: URL of the RapidAPI endpoint
        :param headers_extra: Additional headers if required by the API
        :return: Dictionary of headers
        """
        headers = {
            'x-rapidapi-key': self.api_key,
            'x-rapidapi-host': api_url.split('/')[2]
        }

        if headers_extra:
            headers.update(headers_extra)

        return headers

    def _check_response(self, response: requests.Response) -> Any:
        """
        Check the response status and return the JSON data if successful.

        :param response: Response object from requests library.
        :return: JSON response from the API
        """
        if response.status_code in [200, 201, 202, 204]:
            return response.json() if response.text else None
        response.raise_for_status()

    def call_api(self, api_url: str, method: str = 'GET', headers_extra: Optional[Dict[str, str]] = None, params: Optional[Dict[str, str]] = None, data: Optional[Dict[str, str]] = None, json: Optional[Dict[str, Any]] = None, max_retries: int = 3, backoff_factor: int = 2) -> Any:
        """
        Make a synchronous API call to a RapidAPI endpoint.

        :param api_url: URL of the RapidAPI endpoint
        :param method: HTTP method ('GET' or 'POST')
        :param headers_extra: Additional headers if required by the API
        :param params: Query parameters for GET request
        :param data: Form data for POST request
        :param json: JSON data for POST request
        :param max_retries: Maximum number of retries
        :param backoff_factor: Factor by which the delay increases during each retry
        :return: JSON response from the API
        """
        headers = self._construct_headers(api_url, headers_extra)
        retries = 0

        while retries < max_retries:
            try:
                with requests.request(method, api_url, headers=headers, params=params, data=data, json=json, timeout=self.timeout) as response:
                    return self._check_response(response)
            except requests.RequestException as e:
                retries += 1
                if retries >= max_retries:
                    raise e
                time.sleep(backoff_factor ** retries)

    async def call_api_async(self, api_url: str, method: str = 'GET', headers_extra: Optional[Dict[str, str]] = None, params: Optional[Dict[str, str]] = None, data: Optional[Dict[str, str]] = None, json: Optional[Dict[str, Any]] = None, max_retries: int = 3, backoff_factor: int = 2) -> Any:
        """
        Make an asynchronous API call to a RapidAPI endpoint.

        :param api_url: URL of the RapidAPI endpoint
        :param method: HTTP method ('GET' or 'POST')
        :param headers_extra: Additional headers if required by the API
        :param params: Query parameters for GET request
        :param data: Form data for POST request
        :param json: JSON data for POST request
        :param max_retries: Maximum number of retries
        :param backoff_factor: Factor by which the delay increases during each retry
        :return: JSON response from the API
        """
        headers = self._construct_headers(api_url, headers_extra)

        async with aiohttp.ClientSession() as session:
            retries = 0
            while retries < max_retries:
                try:
                    async with session.request(method, api_url, headers=headers, params=params, data=data, json=json, timeout=self.timeout) as response:
                        if response.status in [200, 201, 202, 204]:
                            return await response.json() if response.text else None
                        response.raise_for_status()
                except aiohttp.ClientError as e:
                    retries += 1
                    if retries >= max_retries:
                        raise e
                    await asyncio.sleep(backoff_factor ** retries)
