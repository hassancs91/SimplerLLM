import os
import time
import asyncio
import aiohttp
import requests
from dotenv import load_dotenv
from typing import Optional, Any, Dict


load_dotenv(override=True) 

class ApifyAPIClient:
    def __init__(self, api_key: Optional[str] = None, timeout: int = 600):
        """
        Initialize the Apify API client.

        :param api_key: Optional API key. If not provided, it will be read from the environment variable 'APIFY_API_KEY'.
        :param timeout: Request timeout in seconds, default is 600 (10 minutes) to accommodate longer actor runs.
        """
        self.api_key = api_key if api_key else os.getenv('APIFY_API_KEY')
        self.timeout = timeout
        self.base_url = "https://api.apify.com/v2"

        if not self.api_key:
            raise ValueError("API key must be provided or set as an environment variable 'APIFY_API_KEY'")

    def _construct_headers(self, headers_extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """
        Construct headers for the Apify API call.

        :param headers_extra: Additional headers if required
        :return: Dictionary of headers
        """
        headers = {
            'Content-Type': 'application/json'
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

    def call_api(self, actor_id: str, actor_input: Dict[str, Any], poll_interval: int = 3) -> Any:
        """
        Run an Apify actor and retrieve its results.

        :param actor_id: Unique identifier for the Apify actor
        :param actor_input: Input parameters for the actor
        :param poll_interval: Time between status checks in seconds
        :return: Actor run results
        """
        # Start actor run
        start_run_url = f"{self.base_url}/acts/{actor_id}/runs"
        params = {"token": self.api_key}
        headers = self._construct_headers()

        try:
            # Start the actor run
            start_response = requests.post(
                start_run_url, 
                json=actor_input, 
                headers=headers, 
                params=params, 
                timeout=self.timeout
            )
            run_data = self._check_response(start_response)
            run_id = run_data.get("data", {}).get("id")

            if not run_id:
                raise ValueError("Failed to retrieve run ID from Apify response")

            # Poll for run status
            total_wait_time = 0
            while total_wait_time < self.timeout:
                status_url = f"{self.base_url}/actor-runs/{run_id}"
                status_response = requests.get(
                    status_url, 
                    params={"token": self.api_key}, 
                    headers=headers
                )
                status_data = self._check_response(status_response)
                status = status_data.get("data", {}).get("status")

                if status in ["SUCCEEDED", "FAILED", "ABORTED"]:
                    break

                time.sleep(poll_interval)
                total_wait_time += poll_interval

            if total_wait_time >= self.timeout:
                raise TimeoutError(f"Actor run exceeded maximum timeout of {self.timeout} seconds")

            if status != "SUCCEEDED":
                raise RuntimeError(f"Actor run failed with status: {status}")

            # Retrieve dataset
            dataset_id = status_data.get("data", {}).get("defaultDatasetId")
            if not dataset_id:
                raise ValueError("Dataset ID not found in the run result")

            dataset_url = f"{self.base_url}/datasets/{dataset_id}/items"
            dataset_response = requests.get(
                dataset_url, 
                params={"token": self.api_key}, 
                headers=headers
            )
            
            return self._check_response(dataset_response)

        except requests.RequestException as e:
            raise RuntimeError(f"API request failed: {str(e)}")
        
    async def call_api_async(self, actor_id: str, actor_input: Dict[str, Any], poll_interval: int = 3) -> Any:
        """
        Asynchronously run an Apify actor and retrieve its results.

        :param actor_id: Unique identifier for the Apify actor
        :param actor_input: Input parameters for the actor
        :param poll_interval: Time between status checks in seconds
        :return: Actor run results
        """
        async with aiohttp.ClientSession() as session:
            # Start actor run
            start_run_url = f"{self.base_url}/acts/{actor_id}/runs"
            headers = self._construct_headers()
            params = {"token": self.api_key}

            try:
                # Start the actor run
                async with session.post(
                    start_run_url, 
                    json=actor_input, 
                    headers=headers, 
                    params=params
                ) as start_response:
                    run_data = await start_response.json() if start_response.status in [200, 201, 202] else None
                    run_id = run_data.get("data", {}).get("id")

                    if not run_id:
                        raise ValueError("Failed to retrieve run ID from Apify response")

                    # Poll for run status
                    total_wait_time = 0
                    while total_wait_time < self.timeout:
                        status_url = f"{self.base_url}/actor-runs/{run_id}"
                        async with session.get(
                            status_url, 
                            params={"token": self.api_key}, 
                            headers=headers
                        ) as status_response:
                            status_data = await status_response.json() if status_response.status in [200, 201, 202] else None
                            status = status_data.get("data", {}).get("status")

                        if status in ["SUCCEEDED", "FAILED", "ABORTED"]:
                            break

                        await asyncio.sleep(poll_interval)
                        total_wait_time += poll_interval

                    if total_wait_time >= self.timeout:
                        raise TimeoutError(f"Actor run exceeded maximum timeout of {self.timeout} seconds")

                    if status != "SUCCEEDED":
                        raise RuntimeError(f"Actor run failed with status: {status}")

                    # Retrieve dataset
                    dataset_id = status_data.get("data", {}).get("defaultDatasetId")
                    if not dataset_id:
                        raise ValueError("Dataset ID not found in the run result")

                    dataset_url = f"{self.base_url}/datasets/{dataset_id}/items"
                    async with session.get(
                        dataset_url, 
                        params={"token": self.api_key}, 
                        headers=headers
                    ) as dataset_response:
                        return await dataset_response.json() if dataset_response.status in [200, 201, 202] else None

            except aiohttp.ClientError as e:
                raise RuntimeError(f"API request failed: {str(e)}")

"""
if __name__ == "__main__":
    
    # Synchronous example
    client = ApifyAPIClient()
    actor_input = {
        "queries": ["apify"],
        "maxResultsPerQuery": 2,
    }
    
    try:
        results = client.call_api("tnudF2IxzORPhg4r8", actor_input)
        for item in results:
            print(item)
    except Exception as err:
        print(f"Error: {err}")


    # Asynchronous example
    async def async_main():
        client = ApifyAPIClient()
        actor_input = {
            "queries": ["apify"],
            "maxResultsPerQuery": 2,
        }

        try:
            results = await client.call_api_async("tnudF2IxzORPhg4r8", actor_input)
            for item in results:
                print(item)
        except Exception as err:
            print(f"Error: {err}")

    asyncio.run(async_main())
"""