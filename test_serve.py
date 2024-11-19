from fastapi.testclient import TestClient
from serve import app  # Import the FastAPI app from serve.py

# Initialize the TestClient with the FastAPI app
client = TestClient(app)


def test_predict_veracity():
    """
    Test the /claim/v1/predict endpoint with a valid input.
    """
    # Define the input payload
    payload = {
        "claim": "Eating garlic prevents COVID-19.",
        "main_text": "Garlic is healthy but there is no evidence that it prevents COVID-19."
    }

    # Send a POST request to the API
    response = client.post("/claim/v1/predict", json=payload)

    # Assertions
    assert response.status_code == 200  # Check if the response is successful
    result = response.json()  # Parse the JSON response

    # Check if the response contains the required fields
    assert "veracity" in result
    assert "confidence" in result

    # Check if veracity is one of the expected labels
    assert result["veracity"] in ["true", "false", "unproven", "mixture"]

    # Check if confidence is a float between 0 and 1
    assert isinstance(result["confidence"], float)
    assert 0.0 <= result["confidence"] <= 1.0


def test_missing_field():
    """
    Test the /claim/v1/predict endpoint with a missing field in the input.
    """
    # Define the input payload with a missing field
    payload = {
        "claim": "Eating garlic prevents COVID-19."
        # Missing "main_text"
    }

    # Send a POST request to the API
    response = client.post("/claim/v1/predict", json=payload)

    # Assertions
    assert response.status_code == 422  # Unprocessable Entity
    assert "detail" in response.json()  # Check for validation error details


def test_empty_input():
    """
    Test the /claim/v1/predict endpoint with an empty input.
    """
    # Define an empty payload
    payload = {}

    # Send a POST request to the API
    response = client.post("/claim/v1/predict", json=payload)

    # Assertions
    assert response.status_code == 422  # Unprocessable Entity
    assert "detail" in response.json()  # Check for validation error details


def test_invalid_data_type():
    """
    Test the /claim/v1/predict endpoint with an invalid data type.
    """
    # Define a payload with incorrect data types
    payload = {
        "claim": 12345,  # Should be a string
        "main_text": True  # Should be a string
    }

    # Send a POST request to the API
    response = client.post("/claim/v1/predict", json=payload)

    # Assertions
    assert response.status_code == 422  # Unprocessable Entity
    assert "detail" in response.json()  # Check for validation error details