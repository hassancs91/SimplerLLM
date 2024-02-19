from  SimplerLLM.tools.rapid_api import RapidAPIClient
import asyncio

api_url = "https://youtube-v2.p.rapidapi.com/search/"
api_params = {
    'query': 'seo',
}


async def main1():
    api_client = RapidAPIClient()  # API key read from environment variable
    response = await api_client.call_api_async(api_url, method='GET', params=api_params)
    print(response)

asyncio.run(main1())


# async def main():
#     try:
#         api_client = RapidAPIClient()
#         response = await api_client.call_api_async(api_url,params=api_params)
#         print("Response:", response)
#     except Exception as e:
#         print("Error:", e)

# if __name__ == "__main__":
#     main()

# asyncio.run(main())

# Example usage
#api_client = RapidAPIClient(api_key='your_manual_api_key') # Manually setting the API key
#api_client = RapidAPIClient()  # API key read from environment variable
#response = api_client.call_api(api_url, method='GET', params=api_params)
#print(response)


