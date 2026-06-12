
import ollama

def ask_llm(prompt):
    """
    Query the codellama model via ollama for code analysis and reasoning.
    
    Args:
        prompt: The prompt containing the question and code context
        
    Returns:
        The model's response as a string
    """
    try:
        response = ollama.generate(
            model="codellama",
            prompt=prompt,
            stream=False
        )
        return response["response"]
    except Exception as e:
        return f"Error querying ollama: {str(e)}\n\nMake sure ollama is running and codellama model is installed.\nRun: ollama pull codellama"
