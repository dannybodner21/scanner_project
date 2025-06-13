import json
import requests

with open("path/to/your/service_account.json") as f:
    service_account_info = json.load(f)

from google.oauth2 import service_account
from google.auth.transport.requests import Request

credentials = service_account.Credentials.from_service_account_info(
    service_account_info, scopes=["https://www.googleapis.com/auth/cloud-platform"]
)
credentials.refresh(Request())
jwt_token = credentials.token

vertex_url = "https://us-central1-aiplatform.googleapis.com/v1/projects/healthy-mark-446922-p8/locations/us-central1/endpoints/1878894947566878720:predict"

payload = {
  "instances": [
    [
      4.02, 4.03, 4.01, 4.02, 49208.11,
      4.023, 4.02335, 4.02326, 4.02441, 0,
      49.94, -5.44e-10, -1.40e-9, 50.0, 50.0,
      4.04889, 4.02335, 3.99409,
      22.44, 0.01514, 0.01225, 0.0, -0.01247,
      0.0, 0.02629, 0.06073,
      1.00324, 0.49999, 0.0052, 0.49999,
      0.00109, 0.00129, -3.33e-10, -3.81e-10, 6.94e-11,
      -0.01507, -0.00230, 0.00186,
      4.93459, 14108.15, -0.02054, 1.0,
      0.0, 0.0, 0.0, 0.0,
      0.00349, 0.99997
    ]
  ]
}

headers = {
    "Authorization": f"Bearer {jwt_token}",
    "Content-Type": "application/json"
}

response = requests.post(vertex_url, headers=headers, json=payload)
response.raise_for_status()
print(response.json())
