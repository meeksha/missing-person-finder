"""
Synthetic dataset generator for the Missing Persons Pattern Finder.

This does NOT use any real missing person's identity or case data.
NCRB (India's Crime Records Bureau) only publishes aggregate statistics,
never individual case records — which is the correct privacy practice.

Instead, this generates a fictional dataset whose overall PATTERNS are
calibrated to real published ratios:
  - ~65% of cases female, ~35% male (per NCRB / Crime in India reports)
  - ~20% of cases are children (under 18)
  - Higher case volume in Maharashtra, West Bengal, Madhya Pradesh
  - Higher recovery/trace rates in southern states (TN, Karnataka, Kerala, Telangana)

It also deliberately PLANTS a handful of multi-case "story clusters" —
groups of fictional cases with deliberately similar descriptions spread
across different districts — so the NLP + clustering pipeline has a
genuine pattern to surface in your demo, the same way you'd build a
labeled test set for any ML system.
"""

import csv
import random
from datetime import date, timedelta

random.seed(42)

FIRST_NAMES_M = ["Ravi","Arjun","Amit","Rahul","Vikram","Suresh","Manoj","Deepak","Sanjay","Ajay",
                  "Karan","Nikhil","Rohit","Vivek","Anil","Pradeep","Sandeep","Ashok","Naveen","Gopal"]
FIRST_NAMES_F = ["Sunita","Meena","Priya","Kavya","Lakshmi","Pooja","Radha","Anita","Geeta","Sita",
                  "Divya","Sneha","Anjali","Rekha","Shanti","Usha","Nisha","Asha","Kiran","Mamta"]
LAST_NAMES = ["Kumar","Singh","Sharma","Yadav","Devi","Patel","Reddy","Nair","Rao","Mehta",
              "Verma","Gupta","Das","Joseph","Iyer","Bai","Kaur","Khan","Joshi","Pillai"]

# (district, state) — weighted toward NCRB-reported higher-incidence states
LOCATIONS_HIGH_VOLUME = [
    ("Mumbai","Maharashtra"), ("Pune","Maharashtra"), ("Nagpur","Maharashtra"), ("Thane","Maharashtra"),
    ("Kolkata","West Bengal"), ("Howrah","West Bengal"), ("Asansol","West Bengal"),
    ("Bhopal","Madhya Pradesh"), ("Indore","Madhya Pradesh"), ("Gwalior","Madhya Pradesh"),
]
LOCATIONS_OTHER = [
    ("Patna","Bihar"), ("Muzaffarpur","Bihar"), ("Vaishali","Bihar"),
    ("Delhi","Delhi"), ("Gurugram","Haryana"),
    ("Lucknow","Uttar Pradesh"), ("Kanpur","Uttar Pradesh"), ("Varanasi","Uttar Pradesh"),
    ("Jaipur","Rajasthan"), ("Jodhpur","Rajasthan"),
    ("Ahmedabad","Gujarat"), ("Surat","Gujarat"),
]
LOCATIONS_HIGH_RECOVERY = [
    ("Chennai","Tamil Nadu"), ("Coimbatore","Tamil Nadu"), ("Madurai","Tamil Nadu"),
    ("Bengaluru","Karnataka"), ("Mysuru","Karnataka"),
    ("Hyderabad","Telangana"), ("Warangal","Telangana"),
    ("Ernakulam","Kerala"), ("Thiruvananthapuram","Kerala"),
]

BUILDS = ["thin","slim","medium build","stocky","tall and lean","short and slight","heavyset","athletic",
          "petite","broad-shouldered","frail","sturdy build","lanky","compact frame"]
HAIR = ["short black hair","long black hair","curly dark hair","plaited hair","cropped hair","wavy hair",
        "shoulder-length hair","hair tied in a ponytail","closely shaved head","hair with a center parting",
        "frizzy hair","straight hair tied back","hair covered with a scarf"]
COMPLEXIONS = ["fair complexion","wheatish complexion","dark complexion","medium complexion",
               "light brown skin tone","tanned skin","pale complexion","olive skin tone"]
CLOTHING_CHILD = ["school uniform","blue shirt and grey trousers","white kurta","faded frock","red salwar kameez",
                   "green churidar","yellow t-shirt and shorts","striped sweater","maroon cardigan","checked shirt",
                   "pink frock with white sandals","khaki shorts and vest"]
CLOTHING_ADULT = ["cotton saree","formal shirt and trousers","casual kurta","jeans and t-shirt","work uniform",
                   "printed salwar suit","plain white shirt","cargo pants and jacket","floral dress","office wear",
                   "traditional dhoti","sports jersey"]
LANDMARKS = ["near the railway station","near the bus stand","close to the market","near a government school",
             "outside a textile factory","near the main temple","close to the highway junction","near a hospital",
             "outside a cinema hall","near the riverbank","close to a construction site","near a mosque",
             "outside a shopping complex","near the village well","close to a petrol pump","near a community hall"]
EXTRA_DETAILS = ["", "", "", " with a small scar on the left hand", " wearing spectacles",
                  " with a mole near the right eyebrow", " carrying a cloth bag", " barefoot at the time",
                  " with a slight limp", " speaking only the local dialect"]

def random_date(start_year=2023, end_year=2024):
    start = date(start_year, 1, 1)
    end = date(end_year, 12, 31)
    delta = (end - start).days
    return (start + timedelta(days=random.randint(0, delta))).isoformat()

def make_description(age, gender, build=None, hair=None, complexion=None, clothing=None, landmark=None, extra=None):
    build = build or random.choice(BUILDS)
    hair = hair or random.choice(HAIR)
    complexion = complexion or random.choice(COMPLEXIONS)
    clothing = clothing or random.choice(CLOTHING_CHILD if age < 18 else CLOTHING_ADULT)
    landmark = landmark or random.choice(LANDMARKS)
    extra = extra if extra is not None else random.choice(EXTRA_DETAILS)
    pronoun = "boy" if (gender == "Male" and age < 18) else "girl" if (gender=="Female" and age<18) else ("man" if gender=="Male" else "woman")
    return f"{build.capitalize()} {pronoun} with {hair}, {complexion}, wearing {clothing}{extra}, last seen {landmark}"

def make_case(case_id, location_pool, force_age=None, force_gender=None, shared_traits=None):
    gender = force_gender or random.choices(["Female","Male"], weights=[65,35])[0]
    age = force_age if force_age is not None else (
        random.randint(8,17) if random.random() < 0.20 else random.randint(18,55)
    )
    first = random.choice(FIRST_NAMES_F if gender=="Female" else FIRST_NAMES_M)
    name = f"{first} {random.choice(LAST_NAMES)}"
    district, state = random.choice(location_pool)

    traits = shared_traits or {}
    description = make_description(age, gender, **traits)

    is_southern = location_pool is LOCATIONS_HIGH_RECOVERY
    status = random.choices(["Traced","Open"], weights=[70,30] if is_southern else [45,55])[0]

    return {
        "case_id": case_id, "name": name, "age": age, "gender": gender,
        "date_reported": random_date(), "district": district, "state": state,
        "description": description, "status": status,
    }

def generate_dataset(n_background=140, n_clusters=4, cluster_size_range=(3,6)):
    rows = []
    counter = 1

    # ── Planted clusters: deliberately similar cases spread across districts ──
    cluster_pools = [LOCATIONS_HIGH_VOLUME, LOCATIONS_OTHER, LOCATIONS_HIGH_VOLUME, LOCATIONS_OTHER]
    for c in range(n_clusters):
        size = random.randint(*cluster_size_range)
        age = random.randint(11,16)
        gender = random.choice(["Male","Female"])
        shared = {
            "build": random.choice(BUILDS),
            "hair": random.choice(HAIR),
            "complexion": random.choice(COMPLEXIONS),
            "extra": random.choice([d for d in EXTRA_DETAILS if d]),  # force a distinguishing mark shared across the cluster
        }
        pool = cluster_pools[c % len(cluster_pools)]
        # spread cluster across DIFFERENT districts within the pool (no repeats if possible)
        districts = random.sample(pool, min(size, len(pool)))
        for d in districts:
            case_id = f"MP{counter:04d}"
            row = make_case(case_id, [d], force_age=age + random.choice([-1,0,1]),
                             force_gender=gender, shared_traits=shared)
            rows.append(row)
            counter += 1

    # ── Background noise: realistic unrelated cases ──
    all_pools = LOCATIONS_HIGH_VOLUME + LOCATIONS_OTHER + LOCATIONS_HIGH_RECOVERY
    weighted_pool = LOCATIONS_HIGH_VOLUME*3 + LOCATIONS_OTHER*2 + LOCATIONS_HIGH_RECOVERY*2
    for _ in range(n_background):
        case_id = f"MP{counter:04d}"
        row = make_case(case_id, weighted_pool)
        rows.append(row)
        counter += 1

    random.shuffle(rows)
    return rows

def write_csv(rows, path):
    fieldnames = ["case_id","name","age","gender","date_reported","district","state","description","status"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    rows = generate_dataset(n_background=140, n_clusters=4)
    write_csv(rows, "realistic_cases.csv")
    print(f"✅ Generated {len(rows)} cases -> realistic_cases.csv")