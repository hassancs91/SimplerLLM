from SimplerLLM.tools.serp import search_with_duck_duck_go,search_with_duck_duck_go_async
import asyncio

def main():
    json_data = search_with_duck_duck_go("prompt engineering")
    print(json_data[0].Domain)

main()


