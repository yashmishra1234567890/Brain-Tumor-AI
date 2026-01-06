from openai import OpenAI
import os

def get_client():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

def generate_medical_explanation(prediction_data: dict):
    """
    Generates human-friendly medical explanation using OpenRouter.
    """
    client = get_client()
    if not client:
        return {
            "explanation": "AI Assistant is currently offline (API Key missing).",
            "disclaimer": "Please consult a doctor for a professional diagnosis."
        }

    system_prompt = (
        "You are a compassionate medical assistant. "
        "Explain the MRI result clearly to a patient. "
        "Avoid medical jargon where possible. "
        "Do NOT provide a diagnosis. Always advise creating a doctor."
    )

    user_prompt = f"""
    The MRI scan analysis indicates:
    - Condition: {prediction_data.get('prediction')}
    - Confidence: {prediction_data.get('confidence')}%

    Please explain:
    1. What this condition typically means.
    2. Why MRI is useful for detecting it.
    3. General next steps (e.g., seeing a neurologist).
    """

    try:
        response = client.chat.completions.create(
            model="mistralai/mistral-7b-instruct",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            extra_headers={
                "HTTP-Referer": "http://localhost:8501",
                "X-Title": "NeuroScan AI"
            }
        )
        return {
            "explanation": response.choices[0].message.content,
            "disclaimer": "This is an AI-generated explanation, not a medical diagnosis."
        }
    except Exception:
        return {
            "explanation": "Could not generate explanation at this time.",
            "disclaimer": "Please try again later."
        }
