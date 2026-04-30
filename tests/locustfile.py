from locust import HttpUser, task, between
import random


class CreditRiskAPIUser(HttpUser):
    # Simulate users waiting 1 to 3 seconds between requests
    wait_time = between(1, 3)

    @task(1)
    def check_health(self):
        """Simulate pinging the health endpoint."""
        self.client.get("/health")

    @task(5)
    def predict_loan_default(self):
        """Simulate a loan application submission."""
        payload = {
            "AMT_INCOME_TOTAL": random.uniform(50000, 300000),
            "AMT_CREDIT": random.uniform(100000, 1000000),
            "AMT_ANNUITY": random.uniform(10000, 50000),
            "DAYS_EMPLOYED": random.randint(-10000, -100),
            "DAYS_BIRTH": random.randint(-25000, -8000),
            "NAME_CONTRACT_TYPE": random.choice(["Cash loans", "Revolving loans"]),
            "CODE_GENDER": random.choice(["M", "F"]),
            "FLAG_OWN_CAR": random.choice(["Y", "N"]),
            "FLAG_OWN_REALTY": random.choice(["Y", "N"]),
        }

        # Post request and group it under "/predict" in the Locust UI
        with self.client.post(
            "/predict", json=payload, catch_response=True, name="/predict"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed with status code {response.status_code}")
