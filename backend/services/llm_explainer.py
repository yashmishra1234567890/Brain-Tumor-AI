from openai import OpenAI
import os

# Initialize OpenAI client
try:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
except Exception:
    client = None

def generate_medical_explanation(prediction_data: dict):
    """
    Uses OpenRouter (Mistral-7B-Instruct) to explain MRI prediction results.
    This function does NOT diagnose or prescribe treatment.
    """

    system_prompt = (
        "You are a medical AI assistant. "
        "Your role is to explain MRI classification results in simple, clear language. "
        "Do NOT provide a medical diagnosis or treatment plan. "
        "Always include a medical disclaimer."
    )

    user_prompt = f"""
MRI Model Output:
Predicted Condition: {prediction_data.get('prediction')}
Confidence: {prediction_data.get('confidence')}%

Class Probabilities:
{prediction_data.get('all_probabilities')}

Tasks:
1. Explain what the predicted condition generally means (simple language).
2. Explain why MRI scans are used for this condition.
3. Suggest general next steps (e.g., consult a specialist).
4. Add a clear disclaimer that this is NOT a medical diagnosis.
"""

    if not client:
        return {
            "explanation": "AI Explanation unavailable. Please configure the OPENROUTER_API_KEY environment variable.",
            "disclaimer": "System not configured for AI explanations."
        }

    try:
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "http://localhost:8501",   # change later to your deployed URL
                "X-Title": "Brain Tumor AI Assistant",
            },
            model="mistralai/mistral-7b-instruct",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3   # lower = safer, factual output
        )

        return {
            "explanation": completion.choices[0].message.content,
            "disclaimer": "This information is for educational purposes only and is not a medical diagnosis."
        }
    except Exception as e:
        print(f"Error generating explanation: {e}")
        return {
            "explanation": "An error occurred while generating the explanation.",
            "disclaimer": "Please try again later."
        }
