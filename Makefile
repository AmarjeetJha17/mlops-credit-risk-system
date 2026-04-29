.PHONY: build up down logs test-health test-metrics test-predict

# Docker commands
build:
	docker-compose build

up:
	docker-compose up -d

down:
	docker-compose down

logs:
	docker-compose logs -f api

# Testing commands (using curl.exe for Windows compatibility)
test-health:
	curl.exe -X GET http://localhost:8000/health

test-metrics:
	curl.exe -X GET http://localhost:8000/metrics

test-predict:
	curl.exe -X POST http://localhost:8000/predict \
	-H "Content-Type: application/json" \
	-d "{ \"AMT_INCOME_TOTAL\": 150000.0, \"AMT_CREDIT\": 500000.0, \"AMT_ANNUITY\": 25000.0, \"DAYS_EMPLOYED\": -1200, \"DAYS_BIRTH\": -10000, \"NAME_CONTRACT_TYPE\": \"Cash loans\", \"CODE_GENDER\": \"F\", \"FLAG_OWN_CAR\": \"Y\", \"FLAG_OWN_REALTY\": \"Y\" }"