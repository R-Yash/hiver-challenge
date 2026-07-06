import json
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from generate_reply import generator
from dotenv import load_dotenv
load_dotenv()

client = genai.Client()

class SingleEmailEval(BaseModel):
    intent_resolution: int = Field(..., description="Score 1-5. Did it address the core problem?")
    intent_rationale: str = Field(..., description="Reasoning for the intent resolution score.")
    
    sop_compliance: int = Field(..., description="Score 1-5. Did it strictly follow the mandatory rules (Jira, OOO calendar, 2 slots, EOD review)?")
    sop_rationale: str = Field(..., description="Reasoning for the SOP compliance score.")
    
    factual_integrity: int = Field(..., description="Score 1-5. 5 means zero hallucinations. Lower if it made things up.")
    factual_rationale: str = Field(..., description="Reasoning for the factual integrity score.")

def evaluate_response(incoming: str, reference: str, generated: str) -> dict:
    prompt = f"""
    You are an expert Corporate Communications QA Auditor. Your job is to evaluate a generated email reply 
    against the original incoming email and the company's reference/historical policy response.

    Evaluate across three metrics on a strict 1-5 scale:
    1. Intent Resolution: Does the reply solve the user's explicit problem?
    2. SOP Compliance: Did the reply include specific required constraints shown in the reference reply 
       (e.g., directing to the IT Self-Service Portal/Jira, reminding to update the OOO calendar, 
       promising PR reviews by EOD, or providing exactly TWO alternative slots for meetings)?
    3. Factual Integrity: Does the reply contain any hallucinations, fake instructions, or facts not supported 
       by the reference context? (5 = perfect adherence, 1 = completely made up facts).

    YOUR RESPONSE MUST STRICTLY FOLLOW THE SCHEMA, WITHOUT ANY EXTRA TEXT, EXPLANATION and ``JSON WRAPPERS``.

    DATA TO EVALUATE:
    ---
    INCOMING EMAIL: 
    {incoming}
    ---
    REFERENCE REPLY (GROUND TRUTH / POLICY): 
    {reference}
    ---
    GENERATED SUGGESTED REPLY: 
    {generated}
    ---
    """

    response = client.interactions.create(
        model='gemini-3.5-flash',
        input=prompt,
        response_format={
            "type": "text", 
            "mime_type": "application/json",
            "schema": SingleEmailEval.model_json_schema()
        }
        
    )

    return SingleEmailEval.model_validate_json(response.output_text).model_dump(mode='json')

if __name__ == "__main__":
    test_suite = [
        {
            "category": "IT Support",
            "incoming": "Hey team, my monitor just went completely black and won't turn back on. I tried swapping cables. Can someone from IT replace this?"
        },
        {
            "category": "HR / PTO",
            "incoming": "Hi team, I'm feeling quite under the weather today and need to take a sudden sick day. I'll check slack intermittently but won't be online for code reviews."
        },
        {
            "category": "Meeting Coordination",
            "incoming": "Hey, I can't make it to our sync at 2 PM today because a production deployment ran over. Can we move it to later today or tomorrow morning?"
        }
    ]

    results = []

    print("=" * 60)

    for i, test in enumerate(test_suite, 1):
        print(f"\n[Test Case {i}] Category: {test['category']}")
        print(f"Incoming: \"{test['incoming']}\"")
        
        response = generator(test['incoming'])
        generated_text = response.response
        
        reference_text = response.source_nodes[0].metadata.get("reference_reply", "No reference found.")

        eval_dict = evaluate_response(test['incoming'], reference_text, generated_text)
        
        results.append({
            "scores": eval_dict
        })

        print(f" -> Intent Resolution: {eval_dict['intent_resolution']}/5 | {eval_dict['intent_rationale']}")
        print(f" -> SOP Compliance:    {eval_dict['sop_compliance']}/5 | {eval_dict['sop_rationale']}")
        print(f" -> Factual Integrity: {eval_dict['factual_integrity']}/5 | {eval_dict['factual_rationale']}")
        print("-" * 60)

    avg_intent = sum(r["scores"]["intent_resolution"] for r in results) / len(results)
    avg_sop = sum(r["scores"]["sop_compliance"] for r in results) / len(results)
    avg_fact = sum(r["scores"]["factual_integrity"] for r in results) / len(results)
    overall_system_score = (avg_intent + avg_sop + avg_fact) / 3

    print("\n" + "="*20 + " SYSTEM PERFORMANCE DASHBOARD " + "="*20)
    print(f"Total Test Cases Processed: {len(results)}")
    print(f"Average Intent Resolution Score: {avg_intent:.2f} / 5.0")
    print(f"Average SOP Compliance Score:    {avg_sop:.2f} / 5.0")
    print(f"Average Factual Integrity Score: {avg_fact:.2f} / 5.0")
    print("-" * 70)
    print(f" OVERALL SYSTEM QUALITY SCORE:  {overall_system_score:.2f} / 5.0")
    print("=" * 70)