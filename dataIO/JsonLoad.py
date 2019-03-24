
import requests

url='https://api.github.com/'

r=requests.get(url)
json_obj=r.json()
print(json_obj)

repos=set()
for entry in json_obj:
    try:
        print(entry)
        repos.add(entry)
    except KeyError as e:
        print("No Key {0}. Skipping...".format(e))

from pprint import pprint
pprint(repos)
