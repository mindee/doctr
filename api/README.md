# Template for your OCR API using docTR

## Installation

You will only need to install [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git), [Docker](https://docs.docker.com/get-docker/) and [poetry](https://python-poetry.org/docs/#installation). The container environment will be self-sufficient and install the remaining dependencies on its own.

## Usage

### Starting your web server

You will need to clone the repository first, go into `api` folder and start the api:

```shell
git clone https://github.com/mindee/doctr.git
cd doctr/api
make run
```

Once completed, your [FastAPI](https://fastapi.tiangolo.com/) server should be running on port 8080.

### Documentation and swagger

FastAPI comes with many advantages including speed and OpenAPI features. For instance, once your server is running, you can access the automatically built documentation and swagger in your browser at: [http://localhost:8080/docs](http://localhost:8080/docs)

### Using the routes

You will find detailed instructions in the live documentation when your server is up, but here are some examples to use your available API routes:

#### Text detection

Using the following image:
<img src="https://user-images.githubusercontent.com/76527547/117319856-fc35bf00-ae8b-11eb-9b51-ca5aba673466.jpg" width="50%" height="50%">

with this snippet:

```python
import requests
with open('/path/to/your/img.jpg', 'rb') as f:
    data = f.read()
print(requests.post("http://localhost:8080/detection", files={'file': data}).json())
```

should yield

```json
[{'box': [0.826171875, 0.185546875, 0.90234375, 0.201171875]},
 {'box': [0.75390625, 0.185546875, 0.8173828125, 0.201171875]}]
```

#### Text recognition

Using the following image:
![recognition-sample](https://user-images.githubusercontent.com/76527547/117133599-c073fa00-ada4-11eb-831b-412de4d28341.jpeg)

with this snippet:

```python
import requests
with open('/path/to/your/img.jpg', 'rb') as f:
    data = f.read()
print(requests.post("http://localhost:8080/recognition", files={'file': data}).json())
```

should yield

```json
{'value': 'invite'}
```

#### End-to-end OCR

Using the following image:
<img src="https://user-images.githubusercontent.com/76527547/117319856-fc35bf00-ae8b-11eb-9b51-ca5aba673466.jpg" width="50%" height="50%">

with this snippet:

```python
import requests
with open('/path/to/your/img.jpg', 'rb') as f:
    data = f.read()
print(requests.post("http://localhost:8080/ocr", files={'file': data}).json())
```

should yield

```json
[{'box': [0.75390625, 0.185546875, 0.8173828125, 0.201171875],
  'value': 'Hello'},
 {'box': [0.826171875, 0.185546875, 0.90234375, 0.201171875],
  'value': 'world!'}]
```
