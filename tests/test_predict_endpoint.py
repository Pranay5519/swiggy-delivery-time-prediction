import pytest
import requests

BASE_URL = "http://127.0.0.1:8000"


@pytest.mark.integration
def test_predict_endpoint():

    payload = {
        "ID": "TEST_1",
        "Delivery_person_ID": "DP_1",
        "Delivery_person_Age": "29",
        "Delivery_person_Ratings": "4.6",
        "Restaurant_latitude": 21.14,
        "Restaurant_longitude": 79.08,
        "Delivery_location_latitude": 21.12,
        "Delivery_location_longitude": 79.05,
        "Order_Date": "15-03-2026",
        "Time_Orderd": "09:15",
        "Time_Order_picked": "09:30",
        "Weatherconditions": "Cloudy",
        "Road_traffic_density": "Low",
        "Vehicle_condition": 0,
        "Type_of_order": "Meal",
        "Type_of_vehicle": "Motorcycle",
        "multiple_deliveries": "1",
        "Festival": "No",
        "City": "Metropolitian"
    }

    url = f"{BASE_URL}/predict"

    response = requests.post(url, json=payload)

    assert response.status_code == 200

    prediction = response.json()

    # your endpoint returns single value
    assert isinstance(prediction, (float, int))