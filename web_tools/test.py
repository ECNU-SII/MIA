import requests
import json

# 测试单个查询
print("Testing single query...")
response = requests.get("http://localhost:8002/search", params={
    "q": "Python programming",
    "max_results": 2,
    "engine": "serper"
})

print("Status code:", response.status_code)
if response.status_code == 200:
    result = response.json()
    print("Success:", result.get("result").get("success"))
    print("Engine:", result.get("result").get("engine"))
    print("Result type:", type(result.get("result")))
    print("Result:", result.get("result").get("results"))
    print("lat")
    print("Sample of result:", str(result.get("result"))[:200] + "..." if len(str(result.get("result"))) > 200 else str(result.get("result")))
else:
    print("Error:", response.text)

print("\n" + "="*50 + "\n")

# 测试数组查询
print("Testing array query...")
response = requests.get("http://localhost:8002/search", params={
    "q": ["Python programming", "Machine learning"],
    "max_results": 2
})

print("Status code:", response.status_code)
if response.status_code == 200:
    result = response.json()
    print("Success:", result.get("result").get("success"))
    print("Engine:", result.get("result").get("engine"))
    print("Result:", result.get("result").get("results"))

else:
    print("Error:", response.text)