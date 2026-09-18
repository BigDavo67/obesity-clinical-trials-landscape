import json
from pathlib import Path
from datetime import datetime, timezone

import requests
import pandas as pd

# ClinicalTrials.gov API endpoint
url = "https://clinicaltrials.gov/api/v2/studies"

# Search for studies where the condition is obesity, 
# return 3 studies, and give the results to me in 
# JSON format.
params = {
    "query.term": (
        "(AREA[Condition]Obesity OR AREA[Condition]Overweight) AND "
        "AREA[StudyType]INTERVENTIONAL AND "
        "(AREA[InterventionType]DRUG OR AREA[InterventionType]BIOLOGICAL)"
    ),
    "pageSize": 1000,
    "format": "json",
    "countTotal": "true"
}

all_studies = []
next_page_token = None
page_number = 1

while True:

    # Add the page token only after the first request
    if next_page_token:
        params["pageToken"] = next_page_token

    # Request one page of studies
    response = requests.get(url, params=params, timeout=30)

    # Raise an error if the request was unsuccessful
    response.raise_for_status()

    # Convert the JSON response into Python
    data = response.json()

    # Get the studies from this page
    page_studies = data["studies"]

    # Add them to our growing list
    all_studies.extend(page_studies)

    print(
        f"Page {page_number}: "
        f"{len(page_studies)} studies retrieved "
        f"({len(all_studies)} total)"
    )

    # Look for the token needed to get the next page
    next_page_token = data.get("nextPageToken")

    # If there is no next-page token, we have reached the end
    if not next_page_token:
        break

    page_number += 1

print()
print("Total studies retrieved:", len(all_studies))

# Extract every study's unique NCT ID
nct_ids = [
    study["protocolSection"]["identificationModule"]["nctId"]
    for study in all_studies
]

# Count unique NCT IDs
unique_nct_ids = set(nct_ids)

print("Unique NCT IDs:", len(unique_nct_ids))
print("Duplicate studies:", len(nct_ids) - len(unique_nct_ids))


# Create an empty list for our trial-level rows
trial_rows = []

for study in all_studies:

    protocol = study["protocolSection"]

    # Access the different sections of the ClinicalTrials.gov record
    identification = protocol.get("identificationModule", {})
    status = protocol.get("statusModule", {})
    design = protocol.get("designModule", {})
    sponsors = protocol.get("sponsorCollaboratorsModule", {})
    conditions_module = protocol.get("conditionsModule", {})
    eligibility = protocol.get("eligibilityModule", {})
    description = protocol.get("descriptionModule", {})

    # Extract individual fields
    nct_id = identification.get("nctId")
    brief_title = identification.get("briefTitle")
    official_title = identification.get("officialTitle")

    phases = design.get("phases", [])
    overall_status = status.get("overallStatus")
    design_info = design.get("designInfo", {})

    primary_purpose = design_info.get("primaryPurpose")
    allocation = design_info.get("allocation")
    intervention_model = design_info.get("interventionModel")

    start_date_struct = status.get("startDateStruct", {})
    start_date = start_date_struct.get("date")
    
    masking_info = design_info.get("maskingInfo", {})
    masking = masking_info.get("masking")
    
    sex = eligibility.get("sex")
    minimum_age = eligibility.get("minimumAge")
    maximum_age = eligibility.get("maximumAge")
    healthy_volunteers = eligibility.get("healthyVolunteers")
    brief_summary = description.get("briefSummary")
    
    primary_completion_date_struct = status.get(
    "primaryCompletionDateStruct", {}
    )
    primary_completion_date = primary_completion_date_struct.get("date")

    completion_date_struct = status.get(
        "completionDateStruct", {}
    )
    completion_date = completion_date_struct.get("date")

    first_posted_date_struct = status.get(
        "studyFirstPostDateStruct", {}
    )
    first_posted_date = first_posted_date_struct.get("date")

    last_update_date_struct = status.get(
        "lastUpdatePostDateStruct", {}
    )
    last_update_date = last_update_date_struct.get("date")
    

    enrollment_info = design.get("enrollmentInfo", {})
    enrollment = enrollment_info.get("count")
    enrollment_type = enrollment_info.get("type")

    lead_sponsor_info = sponsors.get("leadSponsor", {})
    lead_sponsor = lead_sponsor_info.get("name")
    lead_sponsor_class = lead_sponsor_info.get("class")

    conditions = conditions_module.get("conditions", [])

    # Create one simple row representing this trial
    row = {
    "nct_id": nct_id,
    "brief_title": brief_title,
    "official_title": official_title,
    "phase": "; ".join(phases),
    "overall_status": overall_status,
    "start_date": start_date,
    "primary_completion_date": primary_completion_date,
    "completion_date": completion_date,
    "first_posted_date": first_posted_date,
    "last_update_date": last_update_date,
    "enrollment": enrollment,
    "enrollment_type": enrollment_type,
    "lead_sponsor": lead_sponsor,
    "lead_sponsor_class": lead_sponsor_class,
    "conditions": "; ".join(conditions),
    "primary_purpose": primary_purpose,
    "allocation": allocation,
    "intervention_model": intervention_model,
    "masking": masking,
    "sex": sex,
    "minimum_age": minimum_age,
    "maximum_age": maximum_age,
    "healthy_volunteers": healthy_volunteers,
    "brief_summary": brief_summary
}

    trial_rows.append(row)

# Convert the list of rows into a pandas DataFrame
trials_df = pd.DataFrame(trial_rows)

print()
print("Trial-level table created")
print("Rows:", len(trials_df))
print("Columns:", len(trials_df.columns))

print()
print(trials_df.head())

print()
print("Column names:")
print(trials_df.columns.tolist())

print()
print("Missing values by column:")
print(trials_df.isna().sum())

print()
print("Trials with no phase recorded:", (trials_df["phase"] == "").sum())


# Create an empty list for intervention-level rows
intervention_rows = []

for study in all_studies:

    protocol = study["protocolSection"]

    identification = protocol.get("identificationModule", {})
    interventions_module = protocol.get("armsInterventionsModule", {})

    nct_id = identification.get("nctId")

    interventions = interventions_module.get("interventions", [])

    for intervention in interventions:

        intervention_type = intervention.get("type")
        intervention_name = intervention.get("name")
        intervention_description = intervention.get("description")

        other_names = intervention.get("otherNames", [])
        arm_group_labels = intervention.get("armGroupLabels", [])

        row = {
            "nct_id": nct_id,
            "intervention_type": intervention_type,
            "intervention_name": intervention_name,
            "intervention_description": intervention_description,
            "other_names": "; ".join(other_names),
            "arm_group_labels": "; ".join(arm_group_labels)
        }

        intervention_rows.append(row)

# Convert into a pandas DataFrame
interventions_df = pd.DataFrame(intervention_rows)

print()
print("Intervention-level table created")
print("Rows:", len(interventions_df))
print("Columns:", len(interventions_df.columns))

print()
print(interventions_df.head())

print()
print("Intervention types:")
print(interventions_df["intervention_type"].value_counts())

print()
print("Missing values by column:")
print(interventions_df.isna().sum())

print()

unique_intervention_trials = interventions_df["nct_id"].nunique()

print("Unique trials represented in interventions table:",
      unique_intervention_trials)

print(
    "Trials with no intervention rows:",
    len(trials_df) - unique_intervention_trials
)

print(
    "Interventions with no other names:",
    (interventions_df["other_names"] == "").sum()
)

print(
    "Interventions with no arm labels:",
    (interventions_df["arm_group_labels"] == "").sum()
)


# Create an empty list for location-level rows
location_rows = []

for study in all_studies:

    protocol = study["protocolSection"]

    identification = protocol.get("identificationModule", {})
    locations_module = protocol.get("contactsLocationsModule", {})

    nct_id = identification.get("nctId")

    locations = locations_module.get("locations", [])

    for location in locations:

        facility = location.get("facility")
        site_status = location.get("status")
        city = location.get("city")
        state = location.get("state")
        postal_code = location.get("zip")
        country = location.get("country")

        geo_point = location.get("geoPoint", {})
        latitude = geo_point.get("lat")
        longitude = geo_point.get("lon")

        row = {
            "nct_id": nct_id,
            "facility": facility,
            "site_status": site_status,
            "city": city,
            "state": state,
            "postal_code": postal_code,
            "country": country,
            "latitude": latitude,
            "longitude": longitude
        }

        location_rows.append(row)

# Convert into a pandas DataFrame
locations_df = pd.DataFrame(location_rows)

print()
print("Location-level table created")
print("Rows:", len(locations_df))
print("Columns:", len(locations_df.columns))

print()
print(locations_df.head())

print()
print("Missing values by column:")
print(locations_df.isna().sum())

print()

unique_location_trials = locations_df["nct_id"].nunique()

print(
    "Unique trials represented in locations table:",
    unique_location_trials
)

print(
    "Trials with no location rows:",
    len(trials_df) - unique_location_trials
)

print(
    "Unique countries:",
    locations_df["country"].nunique()
)

print()
print("Top 10 countries by number of registered sites:")
print(locations_df["country"].value_counts().head(10))

print()
print(
    "Locations with coordinates:",
    locations_df[["latitude", "longitude"]].notna().all(axis=1).sum()
)

# Create an empty list for collaborator-level rows
collaborator_rows = []

for study in all_studies:

    protocol = study["protocolSection"]

    identification = protocol.get("identificationModule", {})
    sponsors = protocol.get("sponsorCollaboratorsModule", {})

    nct_id = identification.get("nctId")

    collaborators = sponsors.get("collaborators", [])

    for collaborator in collaborators:

        collaborator_name = collaborator.get("name")
        collaborator_class = collaborator.get("class")

        row = {
            "nct_id": nct_id,
            "collaborator_name": collaborator_name,
            "collaborator_class": collaborator_class
        }

        collaborator_rows.append(row)

# Convert into a pandas DataFrame
collaborators_df = pd.DataFrame(collaborator_rows)

print()
print("Collaborator-level table created")
print("Rows:", len(collaborators_df))
print("Columns:", len(collaborators_df.columns))

print()
print(collaborators_df.head())

print()
print("Missing values by column:")
print(collaborators_df.isna().sum())

print()

unique_collaborator_trials = collaborators_df["nct_id"].nunique()

print(
    "Unique trials represented in collaborators table:",
    unique_collaborator_trials
)

print(
    "Trials with no collaborators:",
    len(trials_df) - unique_collaborator_trials
)

print()
print("Top 10 collaborators:")
print(collaborators_df["collaborator_name"].value_counts().head(10))

print()
print("Collaborator classes:")
print(collaborators_df["collaborator_class"].value_counts())

# Define output folders
raw_data_dir = Path("data/raw")
processed_data_dir = Path("data/processed")

# Make sure the folders exist
raw_data_dir.mkdir(parents=True, exist_ok=True)
processed_data_dir.mkdir(parents=True, exist_ok=True)

# Package the untouched API studies with some extraction metadata
raw_export = {
    "metadata": {
        "source": "ClinicalTrials.gov API v2",
        "extracted_at_utc": datetime.now(timezone.utc).isoformat(),
        "query": params["query.term"],
        "candidate_study_count": len(all_studies)
    },
    "studies": all_studies
}

raw_file = raw_data_dir / "obesity_trials_raw.json"

with open(raw_file, "w", encoding="utf-8") as file:
    json.dump(raw_export, file, ensure_ascii=False, indent=2)
    
trials_file = processed_data_dir / "candidate_trials.csv"
interventions_file = processed_data_dir / "candidate_interventions.csv"
locations_file = processed_data_dir / "candidate_locations.csv"
collaborators_file = processed_data_dir / "candidate_collaborators.csv"

trials_df.to_csv(trials_file, index=False)
interventions_df.to_csv(interventions_file, index=False)
locations_df.to_csv(locations_file, index=False)
collaborators_df.to_csv(collaborators_file, index=False)

print()
print("Files saved successfully:")
print(raw_file)
print(trials_file)
print(interventions_file)
print(locations_file)
print(collaborators_file)

# Reload the saved CSV files to confirm they were written correctly
saved_trials_df = pd.read_csv(trials_file)
saved_interventions_df = pd.read_csv(interventions_file)
saved_locations_df = pd.read_csv(locations_file)
saved_collaborators_df = pd.read_csv(collaborators_file)

print()
print("Saved file validation:")
print("Trials:", len(saved_trials_df))
print("Interventions:", len(saved_interventions_df))
print("Locations:", len(saved_locations_df))
print("Collaborators:", len(saved_collaborators_df))

print()
print("=== Pipeline validation ===")

# Create sets of unique NCT IDs from each table
trial_ids = set(trials_df["nct_id"])
intervention_ids = set(interventions_df["nct_id"])
location_ids = set(locations_df["nct_id"])
collaborator_ids = set(collaborators_df["nct_id"])

orphan_intervention_ids = intervention_ids - trial_ids
orphan_location_ids = location_ids - trial_ids
orphan_collaborator_ids = collaborator_ids - trial_ids

print("Intervention NCT IDs missing from trials table:",
      len(orphan_intervention_ids))

print("Location NCT IDs missing from trials table:",
      len(orphan_location_ids))

print("Collaborator NCT IDs missing from trials table:",
      len(orphan_collaborator_ids))

duplicate_trial_rows = trials_df["nct_id"].duplicated().sum()

print("Duplicate NCT IDs in trials table:",
      duplicate_trial_rows)

print(
    "Fully duplicated intervention rows:",
    interventions_df.duplicated().sum()
)

print(
    "Fully duplicated location rows:",
    locations_df.duplicated().sum()
)

print(
    "Fully duplicated collaborator rows:",
    collaborators_df.duplicated().sum()
)

duplicate_locations = locations_df[
    locations_df.duplicated(keep=False)
].sort_values("nct_id")

print()
print("Duplicated location records:")
print(duplicate_locations.to_string(index=False))

print()
print("Phase distribution:")
print(trials_df["phase"].value_counts(dropna=False))

print()
print("Overall status distribution:")
print(trials_df["overall_status"].value_counts(dropna=False))

# Convert date columns into pandas datetime values for validation
start_dates = pd.to_datetime(
    trials_df["start_date"],
    errors="coerce"
)

primary_completion_dates = pd.to_datetime(
    trials_df["primary_completion_date"],
    errors="coerce"
)

completion_dates = pd.to_datetime(
    trials_df["completion_date"],
    errors="coerce"
)

first_posted_dates = pd.to_datetime(
    trials_df["first_posted_date"],
    errors="coerce"
)

last_update_dates = pd.to_datetime(
    trials_df["last_update_date"],
    errors="coerce"
)


# Convert date columns into pandas datetime values.
# format="mixed" allows different levels of date precision,
# such as YYYY-MM-DD, YYYY-MM, or YYYY.
start_dates = pd.to_datetime(
    trials_df["start_date"],
    format="mixed",
    errors="coerce"
)

primary_completion_dates = pd.to_datetime(
    trials_df["primary_completion_date"],
    format="mixed",
    errors="coerce"
)

completion_dates = pd.to_datetime(
    trials_df["completion_date"],
    format="mixed",
    errors="coerce"
)

first_posted_dates = pd.to_datetime(
    trials_df["first_posted_date"],
    format="mixed",
    errors="coerce"
)

last_update_dates = pd.to_datetime(
    trials_df["last_update_date"],
    format="mixed",
    errors="coerce"
)


# Check whether any non-missing date values failed to parse
print()
print("Date parsing failures:")

print(
    "Start date:",
    start_dates.isna().sum()
    - trials_df["start_date"].isna().sum()
)

print(
    "Primary completion date:",
    primary_completion_dates.isna().sum()
    - trials_df["primary_completion_date"].isna().sum()
)

print(
    "Completion date:",
    completion_dates.isna().sum()
    - trials_df["completion_date"].isna().sum()
)

print(
    "First posted date:",
    first_posted_dates.isna().sum()
    - trials_df["first_posted_date"].isna().sum()
)

print(
    "Last update date:",
    last_update_dates.isna().sum()
    - trials_df["last_update_date"].isna().sum()
)


# Check whether the dates occur in a sensible chronological order
primary_before_start = (
    primary_completion_dates.notna()
    & start_dates.notna()
    & (primary_completion_dates < start_dates)
).sum()

completion_before_start = (
    completion_dates.notna()
    & start_dates.notna()
    & (completion_dates < start_dates)
).sum()

completion_before_primary = (
    completion_dates.notna()
    & primary_completion_dates.notna()
    & (completion_dates < primary_completion_dates)
).sum()


print()
print("Date order checks:")

print(
    "Primary completion before start:",
    primary_before_start
)

print(
    "Completion before start:",
    completion_before_start
)

print(
    "Completion before primary completion:",
    completion_before_primary
)