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

headers = {"accept": "application/json"}
params = {"det_arch": "db_resnet50"}

with open('/path/to/your/img.jpg', 'rb') as f:
    files = [  # application/pdf, image/jpeg, image/png supported
        ("files", ("117319856-fc35bf00-ae8b-11eb-9b51-ca5aba673466.jpg", f.read(), "image/jpeg")),
    ]
print(requests.post("http://localhost:8080/detection", headers=headers, params=params, files=files).json())
```

should yield

```json
[
  {
    "name": "117319856-fc35bf00-ae8b-11eb-9b51-ca5aba673466.jpg",
    "geometries": [
      [
        0.8176307908857315,
        0.1787109375,
        0.9101580212741838,
        0.2080078125
      ],
      [
        0.7471996155154171,
        0.1796875,
        0.8272978149561669,
        0.20703125
      ]
    ]
  }
]
```

#### Text recognition

Using the following image:
![recognition-sample](https://user-images.githubusercontent.com/76527547/117133599-c073fa00-ada4-11eb-831b-412de4d28341.jpeg)

with this snippet:

```python
import requests

headers = {"accept": "application/json"}
params = {"reco_arch": "crnn_vgg16_bn"}

with open('/path/to/your/img.jpg', 'rb') as f:
    files = [  # application/pdf, image/jpeg, image/png supported
        ("files", ("117133599-c073fa00-ada4-11eb-831b-412de4d28341.jpeg", f.read(), "image/jpeg")),
    ]
print(requests.post("http://localhost:8080/recognition", headers=headers, params=params, files=files).json())
```

should yield

```json
[
  {
    "name": "117133599-c073fa00-ada4-11eb-831b-412de4d28341.jpeg",
    "value": "invite",
    "confidence": 1.0
  }
]
```

#### End-to-end OCR

Using the following image:
<img src="https://user-images.githubusercontent.com/76527547/117319856-fc35bf00-ae8b-11eb-9b51-ca5aba673466.jpg" width="50%" height="50%">

with this snippet:

```python
import requests

headers = {"accept": "application/json"}
params = {"det_arch": "db_resnet50", "reco_arch": "crnn_vgg16_bn"}

with open('/path/to/your/img.jpg', 'rb') as f:
    files = [  # application/pdf, image/jpeg, image/png supported
        ("files", ("117319856-fc35bf00-ae8b-11eb-9b51-ca5aba673466.jpg", f.read(), "image/jpeg")),
    ]
print(requests.post("http://localhost:8080/ocr", headers=headers, params=params, files=files).json())
```

should yield

```json
[
  {
    "name": "117319856-fc35bf00-ae8b-11eb-9b51-ca5aba673466.jpg",
    "orientation": {
      "value": 0,
      "confidence": null
    },
    "language": {
      "value": null,
      "confidence": null
    },
    "dimensions": [2339, 1654],
    "items": [
      {
        "blocks": [
          {
            "geometry": [
              0.7471996155154171,
              0.1787109375,
              0.9101580212741838,
              0.2080078125
            ],
            "objectness_score": 0.5,
            "lines": [
              {
                "geometry": [
                  0.7471996155154171,
                  0.1787109375,
                  0.9101580212741838,
                  0.2080078125
                ],
                "objectness_score": 0.5,
                "words": [
                  {
                    "value": "Hello",
                    "geometry": [
                      0.7471996155154171,
                      0.1796875,
                      0.8272978149561669,
                      0.20703125
                    ],
                    "objectness_score": 0.5,
                    "confidence": 1.0,
                    "crop_orientation": {"value": 0, "confidence": null}
                  },
                  {
                    "value": "world!",
                    "geometry": [
                      0.8176307908857315,
                      0.1787109375,
                      0.9101580212741838,
                      0.2080078125
                    ],
                    "objectness_score": 0.5,
                    "confidence": 1.0,
                    "crop_orientation": {"value": 0, "confidence": null}
                  }
                ]
              }
            ]
          }
        ]
      }
    ]
  }
]
```
