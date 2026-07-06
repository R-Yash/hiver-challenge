import json
from pydantic import BaseModel
from google import genai
from dotenv import load_dotenv
load_dotenv()

client = genai.Client()

class EmailPair(BaseModel):
    category: str
    incoming_email: str
    reference_reply: str

class EmailDataset(BaseModel):
    emails: list[EmailPair]

def generate_email_batch(batch_size: int = 25) -> list[dict]:
    prompt = f"""
    You are an expert dataset generator. Generate a dataset of {batch_size} realistic email-reply pairs 
    for day-to-day internal communications within a tech company. 
    
    Ensure variety in tone (casual, urgent, formal) and categorize them into: 
    IT Support, HR/PTO, Code Reviews, Meeting Coordination, and General Chatter.

    CRITICAL RULES FOR THE REPLIES (The 'reference_reply' must follow these SOPs):
    1. IT Support: Always ask for the Jira ticket number or direct the user to the IT Self-Service Portal.
    2. HR/PTO: Always remind the user to update the "Team OOO Calendar".
    3. Code Reviews: Always commit to reviewing the PR "by EOD". If a link is missing, ask for it.
    4. Meeting Coordination: If rescheduling, always propose exactly two alternative time slots.
    
    Make the emails look like real corporate communication (e.g., typos, brevity, signatures).
    """

    response = client.interactions.create(
        model='gemini-3.5-flash',
        input=prompt,
        response_format={
            "type": "text", 
            "mime_type": "application/json",
            "schema": EmailDataset.model_json_schema()
        },
        generation_config={
            "temperature": 0.7 
        }
    )
    return EmailDataset.model_validate_json(response.output_text).emails

if __name__ == "__main__":
    final_dataset = []
    
    for i in range(3): 
        print(f"Batch {i+1}...")
        try:
            batch = generate_email_batch(25)
            final_dataset.extend([email.model_dump(mode='json') for email in batch])
        except Exception as e:
            print(e)
            
    with open("synthetic_dataset.json", "w") as f:
        json.dump(final_dataset, f, indent=4)
        