from google.genai import types

def correct_transcript(transcript, client):
    instructions = """
    You are a helpful grammar correction assistant. Given a sentence, correct any grammatical errors and return the corrected version.
    Only provide the revised text without any additional text.
    """
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=transcript,
        config=types.GenerateContentConfig(
            system_instruction=instructions
        )
    )
    
    return response.text.strip()  # Return the corrected transcript without leading/trailing whitespace.