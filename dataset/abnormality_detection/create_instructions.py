import random 
import json
import numpy as np
from collections import defaultdict
from collections import Counter

def format_boxes(bounding_boxes, num_float=2):
    formatted_boxes = [
        f"[{round(bbox[0], num_float)}, {round(bbox[1], num_float)}, {round(bbox[2], num_float)}, {round(bbox[3], num_float)}]" for bbox in bounding_boxes
        ]
    boxes_str = ', '.join(formatted_boxes[:-1]) + ('' if len(formatted_boxes) < 2 else ' and ') + formatted_boxes[-1]
    return boxes_str


def select_article(word):
    if word[0].lower() in 'aeiou':
        return 'an'
    else:
        return 'a'

def generate_instruction_abnormalities_grouped(bounding_boxes, abnormalities):
    """
    Generate a question and an answer about the presence and location of abnormalities
    (or lesions) on a Chest X-ray, incorporating variations for questions and answers and
    correctly using articles ('a' or 'an') before abnormalities. Abnormalities mentioned multiple
    times will be grouped with all their associated bounding boxes.

    Parameters:
    - bounding_boxes (list of lists): A list where each element is a bounding box
      represented as [x1, y1, x2, y2], assumed to be integer values.
    - abnormalities (list of str): A list of abnormality names corresponding to each bounding box.

    Returns:
    - A JSON string containing a dict for the question and the answer.
    """

    question_variations = [
        "Could you indicate if there are any abnormalities on this Chest X-ray and their locations?",
        "Are abnormalities present on this Chest X-ray? Where exactly can they be found?",
        "Please identify any lesions or abnormalities on this X-ray and specify their locations.",
        "On this Chest X-ray, can you point out any abnormalities and their precise positions?",
        "I need information on any abnormalities or lesions on this X-ray, including their locations. Can you help?",
        "Can you detect and describe the location of any abnormalities found on this Chest X-ray?",
        "Are there identifiable abnormalities on this Chest X-ray? If so, where are they located?",
        "Tell me about any lesions on this X-ray and detail their specific locations.",
        "Do any abnormalities appear on this Chest X-ray? Please point them out along with their locations.",
        "Identify any abnormalities or lesions present on this X-ray and provide their exact locations."
    ]

    answer_prefix_variations = [
        "Sure! I can find",
        "Indeed, there are",
        "Yes, the following abnormalities are identified:",
        "Upon examination, I detect",
        "The analysis reveals",
        "After a detailed review, we have discovered",
        "The findings include",
        "Notably, I can identify",
        "Based on the image, there are",
        "From the examination, it's evident there are"
    ]

    # Select a random question variation
    question = random.choice(question_variations)

    no_lesions_answers = [
    "I can't find any lesion on the image.",
    "No abnormalities or lesions are detected on this Chest X-ray.",
    "The Chest X-ray appears to be clear of any lesions or abnormalities.",
    "Upon review, no lesions are visible on the image.",
    "The examination reveals a clean bill of health with no visible abnormalities.",
    "This Chest X-ray shows no signs of abnormalities or lesions.",
    "No lesions or abnormalities are present on this X-ray, as far as I can tell.",
    "After a thorough examination, I conclude that there are no detectable lesions on this X-ray.",
    "The image does not display any abnormalities or lesions.",
    "Based on this X-ray, it appears there are no lesions or abnormalities to report."
    ]

    if not bounding_boxes or not abnormalities:
        answer = random.choice(no_lesions_answers)
    else:
        if len(bounding_boxes) != len(abnormalities):
            raise ValueError("Bounding boxes and abnormalities lists must be of equal length.")

        # Assuming abnormalities and bounding_boxes are already defined lists
        abnormalities_dict = defaultdict(list)
        for abnormality, bbox in zip(abnormalities, bounding_boxes):
            abnormalities_dict[abnormality].append(bbox)

        # Format each abnormality with its associated coordinates and the correct article
        abnormalities_descriptions = []
        for abnormality, boxes in abnormalities_dict.items():
            article = select_article(abnormality)
            abnormality = abnormality.lower()
            boxes_str = format_boxes(boxes)
            abnormalities_descriptions.append(f"{article} {abnormality} located at the coordinates {boxes_str}")

        # Join all abnormalities descriptions
        abnormalities_str = "; ".join(abnormalities_descriptions)

        # Select a random answer prefix variation
        answer_prefix = random.choice(answer_prefix_variations)
        answer = f"{answer_prefix} {abnormalities_str}."

    instruction = {
        "question": question,
        "answer": answer
    }

    return instruction


