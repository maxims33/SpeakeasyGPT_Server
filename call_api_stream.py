import requests

url = "http://127.0.0.1:5000/query_streaming"
headers = {'Content-Type': 'application/json'}
prompt = "Explain quantum mechanics"
data = '{"prompt": "' + prompt + '"}'

print("data: " + data)

def process_chunk(chunk):
    print(chunk.decode('utf-8'), end="", flush=True)

with requests.post(url, data=data, stream=True, headers=headers) as response:
    response.raise_for_status()  # Check for HTTP errors
    for chunk in response.iter_content(chunk_size=None): 
        # Process each chunk as it arrives
        process_chunk(chunk)
