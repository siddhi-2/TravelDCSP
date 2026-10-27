"""
Builds a SMALL fake dataset that has the SAME file layout and column names as the
official TravelPlanner release (queries csv + 6-source database). Used to test the
real-data code path offline.   python tests/make_fixture.py data/fixture
"""
import os, random, sys, csv, itertools

out = sys.argv[1] if len(sys.argv) > 1 else "data/fixture"
r = random.Random(7)
STATES = {"Texas": ["Austin", "Dallas", "Houston", "Plano"], "Ohio": ["Columbus", "Cleveland", "Dayton"],
          "Florida": ["Miami", "Tampa", "Orlando", "Sarasota"]}
CITIES = [c for v in STATES.values() for c in v]
DATES = [f"2022-03-{d:02d}" for d in range(1, 15)]
CUIS = ["Chinese", "American", "Italian", "Mexican", "Indian", "Bakery", "Cafe", "Pizza", "French"]

def w(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        cw = csv.writer(f); cw.writerow(header); cw.writerows(rows)

db = os.path.join(out, "database")
# flights
rows = []
for o, d in itertools.permutations(CITIES, 2):
    for date in DATES:
        for _ in range(r.randint(0, 3)):
            rows.append([f"F{r.randint(1000000,9999999)}", r.randint(60, 600), "10:04", "11:41", 90, date, o, d, 800])
w(f"{db}/flights/clean_Flights_2022.csv",
  ["Flight Number", "Price", "DepTime", "ArrTime", "ActualElapsedTime", "FlightDate", "OriginCityName", "DestCityName", "Distance"], rows)
# distance
rows = [[o, d, "", "", f"{r.randint(80, 2200)} km"] for o, d in itertools.permutations(CITIES, 2)]
w(f"{db}/googleDistanceMatrix/distance.csv", ["origin", "destination", "cost", "duration", "distance"], rows)
# accommodations
RT = ["Entire home/apt", "Private room", "Shared room", "Hotel room"]
RULES = ["No smoking", "No parties", "No pets", "No visitors", "No children under 10"]
rows = []
for c in CITIES:
    for i in range(25):
        rows.append([f"{c} Stay {i}", r.randint(40, 450), r.choice(RT), " & ".join(r.sample(RULES, r.randint(0, 2))),
                     r.randint(1, 4), r.randint(1, 6), round(r.uniform(2, 5), 1), c])
w(f"{db}/accommodations/clean_accommodations_2022.csv",
  ["NAME", "price", "room type", "house_rules", "minimum nights", "maximum occupancy", "review rate number", "city"], rows)
# restaurants
rows = []
for c in CITIES:
    for i in range(60):
        rows.append([f"{c} Eatery {i}", r.randint(8, 90), ", ".join(r.sample(CUIS, r.randint(1, 3))), round(r.uniform(2, 5), 1), c])
w(f"{db}/restaurants/clean_restaurant_2022.csv", ["Name", "Average Cost", "Cuisines", "Aggregate Rating", "City"], rows)
# attractions
rows = [[f"{c} Sight {i}", 0.0, 0.0, "addr", "", "", c] for c in CITIES for i in range(20)]
w(f"{db}/attractions/attractions.csv", ["Name", "Latitude", "Longitude", "Address", "Phone", "Website", "City"], rows)
# background
os.makedirs(f"{db}/background", exist_ok=True)
open(f"{db}/background/citySet.txt", "w").write("\n".join(CITIES))
open(f"{db}/background/citySet_with_states.txt", "w").write("\n".join(f"{c}\t{s}" for s, cs in STATES.items() for c in cs))

# queries (stringified literals, like the HF csv export)
def q(i, org, dest, days, k, people, budget, lc, level):
    dates = DATES[:days]
    text = f"Plan a {days}-day trip from {org} to {dest} for {people} people with a budget of ${budget}."
    return [org, dest, days, k, str(dates), people, str(dict({"house rule": None, "cuisine": None, "room type": None,
            "transportation": None}, **lc)), budget, text, level]
H = ["org", "dest", "days", "visiting_city_number", "date", "people_number", "local_constraint", "budget", "query", "level"]
train = [q(0, "Miami", "Austin", 3, 1, 1, 1800, {}, "easy"),
         q(1, "Tampa", "Ohio", 5, 2, 2, 5200, {"cuisine": ["Chinese", "Italian"]}, "medium"),
         q(2, "Dallas", "Florida", 7, 3, 3, 9000, {"room type": "not shared room", "house rule": "smoking"}, "hard")]
val = [q(3, "Houston", "Columbus", 3, 1, 2, 2500, {"transportation": "no flight"}, "easy"),
       q(4, "Orlando", "Ohio", 5, 2, 3, 6500, {"room type": "entire room", "cuisine": ["Indian"]}, "medium"),
       q(5, "Cleveland", "Texas", 7, 3, 4, 12000, {"transportation": "no self-driving", "house rule": "parties",
                                                    "cuisine": ["French", "Mexican"]}, "hard"),
       q(6, "Austin", "Miami", 3, 1, 5, 900, {}, "easy")]
w(f"{out}/train.csv", H, train)
w(f"{out}/validation.csv", H, val)
print("fixture written to", out)
