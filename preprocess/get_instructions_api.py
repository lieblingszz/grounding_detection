import os
import argparse
import openai
import json
from dotenv import load_dotenv

# Load the API key to generate instructions
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
loaded = load_dotenv(dotenv_path)
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_instructions(task):
    """
    Generate instructions for specific tasks using the ChatGPT API.
    Each category with 1 to 5 placeholders should have exactly five variations,
    with each placeholder represented by a unique, non-specific label like <anatomy1>, <anatomy2>, etc.
    """
    example_sentences = {
        "visual grounding": "Given the image, localize the region of <anatomy>.",
        "image annotation": "Annotate all objects in the image with their respective labels.",
        "data entry": "Transcribe the text from the image into a text format."
    }
    example_prompt = example_sentences.get(task)
    
    if not example_prompt:
        raise ValueError("No example sentence available for the given task.")

    instructions = {}
    placeholder_counts = range(1, 6)  
    
    try:
        for count in placeholder_counts:
            placeholders = ", ".join(f"<anatomy{i}>" for i in range(1, count + 1))
            modified_prompt = example_prompt.replace("<anatomy>", placeholders)
            prompt_text = f"Generate five variations of this sentence, using the placeholders {placeholders} as literal text, not to be substituted with specific anatomical terms: '{modified_prompt}'"
            response = openai.ChatCompletion.create(
                model="gpt-4-turbo",  
                messages=[
                    {"role": "system", "content": "Generate detailed task instructions."},
                    {"role": "user", "content": prompt_text}
                ],
                max_tokens=200 * count,  
                n=1,
                temperature=0.8
            )
            key = f"{count}_anatomy" if count == 1 else f"{count}_anatomies"
            instructions[key] = [
                line.split('. ', 1)[-1].strip().strip('"').strip("'")
                for line in response['choices'][0]['message']['content'].split('\n')
                if line.strip()
            ]

    except Exception as e:
        print(f"An error occurred: {e}")
    
    return instructions

def parse_arguments():
    """ Parse the command line arguments. """
    parser = argparse.ArgumentParser(description="Generate instructions for specific tasks using ChatGPT API.")
    parser.add_argument('task', type=str, help='The type of task for which to generate instructions.')
    return parser.parse_args()

def write_instructions(task, instructions, file_name="instructions.json"):
    """ Write the instructions to a JSON file, ensuring proper formatting. """
    data = {task: instructions}
    with open(file_name, "w") as f:
        json.dump(data, f, indent=4)  

def main():
    args = parse_arguments()
    instructions = generate_instructions(args.task)
    write_instructions(args.task, instructions)

if __name__ == "__main__":
    main()
