import os
import cohere
import guardrails as gd
from guardrails.hub import ValidRange, ValidChoices
from pydantic import BaseModel, Field
from rich import print
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Setup the Cohere client
co = cohere.Client(os.getenv("COHERE_API_KEY")) # Get your free API key: https://dashboard.cohere.com/api-keys

doctors_notes = """49 y/o Male with chronic macular rash to face & hair, worse in beard, eyebrows & nares.
Itchy, flaky, slightly scaly. Moderate response to OTC steroid cream"""

class Symptom(BaseModel):
    symptom: str = Field(..., description="Symptom that a patient is experiencing")
    affected_area: str = Field(
        ...,
        description="What part of the body the symptom is affecting",
        json_schema_extra={"validators": [ValidChoices(["Head", "Face", "Neck", "Chest"], on_fail="reask")]}
    )

class CurrentMed(BaseModel):
    medication: str = Field(..., description="Name of the medication the patient is taking")
    response: str = Field(..., description="How the patient is responding to the medication")


class PatientInfo(BaseModel):
    gender: str = Field(..., description="Patient's gender")
    age: int = Field(..., description="Patient's age", json_schema_extra={"validators": [ValidRange(0, 100)]})
    symptoms: List[Symptom] = Field(..., description="Symptoms that the patient is experiencing")
    current_meds: List[CurrentMed] = Field(..., description="Medications that the patient is currently taking")


PROMPT = """Given the following doctor's notes about a patient,
please extract a dictionary that contains the patient's information.

${doctors_notes}

${gd.complete_json_suffix_v2}
"""

# Initialize a Guard object from the Pydantic model PatientInfo
guard = gd.Guard.from_pydantic(PatientInfo)

# Prepare the prompt with the doctor's notes
prompt_text = f"""Given the following doctor's notes about a patient,
please extract a dictionary that contains the patient's information.

{doctors_notes}
"""

# Wrap the Cohere API call with the `guard` object
response = guard(
    messages=[{"role": "user", "content": prompt_text}],
    model='command-a-03-2025',
    temperature=0,
    num_reasks=3,
)

# Print the validated output from the LLM
print(response.validated_output)