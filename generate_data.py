"""
Generates 50 synthetic patient discharge summaries as .txt files
into the data/raw/ folder. These simulate real hospital documents
with intentional messiness — missing fields, inconsistent formatting —
to make the pipeline realistic.
"""

import os
import random

output_dir = "data/raw"
os.makedirs(output_dir, exist_ok=True)

# Seed data pools
diagnoses = [
    "Type 2 Diabetes Mellitus", "Congestive Heart Failure",
    "COPD exacerbation", "Acute Myocardial Infarction",
    "Pneumonia", "Hip Fracture", "Sepsis",
    "Chronic Kidney Disease Stage 3", "Hypertensive Crisis",
    "Lumbar Spinal Stenosis"
]

medications = [
    "Metformin 500mg", "Lisinopril 10mg", "Atorvastatin 40mg",
    "Furosemide 20mg", "Aspirin 81mg", "Insulin Glargine",
    "Albuterol inhaler", "Warfarin 5mg", "Amlodipine 5mg",
    "Omeprazole 20mg"
]

doctors = [
    "Dr. Sarah Chen", "Dr. Michael Torres", "Dr. Priya Patel",
    "Dr. James Wilson", "Dr. Emily Rodriguez"
]

follow_up_options = [
    "Follow-up appointment scheduled in 7 days with primary care physician.",
    "Patient advised to follow up with cardiologist in 2 weeks.",
    "No follow-up appointment scheduled at time of discharge.",
    "Follow-up in 30 days with specialist.",
    ""  # intentionally missing
]

def generate_summary(patient_id):
    """Generate one synthetic discharge summary with realistic messiness."""
    
    age = random.randint(45, 89)
    gender = random.choice(["Male", "Female"])
    diagnosis = random.choice(diagnoses)
    los = random.randint(1, 14)  # length of stay in days
    prior_admissions = random.randint(0, 5)
    meds = random.sample(medications, random.randint(2, 5))
    doctor = random.choice(doctors)
    follow_up = random.choice(follow_up_options)
    
    # Intentional messiness: sometimes omit fields
    omit_age = random.random() < 0.1
    omit_diagnosis = random.random() < 0.05
    omit_meds = random.random() < 0.08
    
    age_line = f"Age: {age}" if not omit_age else "Age: Unknown"
    diagnosis_line = f"Primary Diagnosis: {diagnosis}" if not omit_diagnosis else "Primary Diagnosis: Not documented"
    meds_line = f"Medications at Discharge: {', '.join(meds)}" if not omit_meds else "Medications at Discharge: See pharmacy notes"
    
    summary = f"""
DISCHARGE SUMMARY
=================
Patient ID: PAT-{patient_id:04d}
{age_line}
Gender: {gender}
Attending Physician: {doctor}

CLINICAL INFORMATION
--------------------
{diagnosis_line}
Length of Stay: {los} days
Prior Admissions (past 12 months): {prior_admissions}

{meds_line}

DISCHARGE NOTES
---------------
Patient was admitted for management of {diagnosis.lower() if not omit_diagnosis else 'undocumented condition'}.
{'Patient showed signs of improvement after treatment.' if random.random() > 0.3 else 'Patient condition stabilized but remains complex.'}
{'Patient has history of non-compliance with medications.' if random.random() < 0.2 else ''}
{'Social support: Patient lives alone.' if random.random() < 0.25 else 'Social support: Family present.'}

{follow_up}

Signed: {doctor}
    """.strip()
    
    return summary, {
        "patient_id": f"PAT-{patient_id:04d}",
        "age": age if not omit_age else None,
        "gender": gender,
        "diagnosis": diagnosis if not omit_diagnosis else None,
        "length_of_stay": los,
        "prior_admissions": prior_admissions,
        "medications": meds if not omit_meds else None,
        "follow_up": follow_up,
        "has_follow_up": bool(follow_up.strip()),
    }

if __name__ == "__main__":
    print("Generating 50 synthetic discharge summaries...")
    for i in range(1, 51):
        summary_text, _ = generate_summary(i)
        filepath = os.path.join(output_dir, f"patient_{i:04d}.txt")
        with open(filepath, "w") as f:
            f.write(summary_text)
    print(f"Done. 50 files written to {output_dir}/")

