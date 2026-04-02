import requests

BASE_URL = "http://127.0.0.1:5000"

# ── Health check ───────────────────────────────────────────────────────────────
print("\n── Health Check ──")
r = requests.get(f"{BASE_URL}/health")
print(r.json())

# ── Valid input ────────────────────────────────────────────────────────────────
print("\n── Valid Prediction ──")
valid_payload = {
    "age"           : 35,
    "job"           : "admin.",
    "marital"       : "married",
    "education"     : "university.degree",
    "housing"       : "yes",
    "loan"          : "no",
    "contact"       : "cellular",
    "month"         : "may",
    "day_of_week"   : "mon",
    "campaign"      : 2,
    "previous"      : 1,
    "poutcome"      : "nonexistent",
    "emp.var.rate"  : -1.8,
    "cons.price.idx": 92.893,
    "cons.conf.idx" : -46.2,
    "euribor3m"     : 1.313,
    "nr.employed"   : 5099.1,
    "previous_contact": 0
}
r = requests.post(f"{BASE_URL}/predict", json=valid_payload)
print(r.json())

# ── Missing field ──────────────────────────────────────────────────────────────
print("\n── Missing Field Test ──")
bad_payload = valid_payload.copy()
del bad_payload['age']
r = requests.post(f"{BASE_URL}/predict", json=bad_payload)
print(r.json())

# ── Wrong type ────────────────────────────────────────────────────────────────
print("\n── Wrong Type Test ──")
bad_payload2 = valid_payload.copy()
bad_payload2['age'] = "thirty five"
r = requests.post(f"{BASE_URL}/predict", json=bad_payload2)
print(r.json())

# ── Out of range ──────────────────────────────────────────────────────────────
print("\n── Out of Range Test ──")
bad_payload3 = valid_payload.copy()
bad_payload3['age'] = 200
r = requests.post(f"{BASE_URL}/predict", json=bad_payload3)
print(r.json())

# ── Invalid category ──────────────────────────────────────────────────────────
print("\n── Invalid Category Test ──")
bad_payload4 = valid_payload.copy()
bad_payload4['job'] = "astronaut"
r = requests.post(f"{BASE_URL}/predict", json=bad_payload4)
print(r.json())