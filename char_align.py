import json
from functools import cmp_to_key
from ast import literal_eval
import pandas as pd
import re
import numpy as np
import xlsxwriter
from itertools import zip_longest
import os

BOX_PATH_PREFIX = 'response/thanh_giao_yeu_ly_image_'
BOX_PATH_SUFFIX = '.txt'

TEXT_PATH_PREFIX = 'processed_text/page_'
TEXT_PATH_SUFFIX = '.txt'

TOP = 96.0
BOTTOM = 618.0
LEFT = 71.0
RIGHT = 394.0

# Load dictionaries
quoc_ngu_sino_nom_df = pd.read_excel("QuocNgu_SinoNom_Dic.xlsx")
sino_nom_similar_df = pd.read_excel("SinoNom_similar_Dic.xlsx")

# Build similar character dictionary from SinoNom_similar_Dic
sino_nom_similar_dict = {}
for _, row in sino_nom_similar_df.iterrows():
    root_char = row['Input Character']
    similar_chars = list(literal_eval(row['Top 20 Similar Characters']))
    sino_nom_similar_dict[root_char] = similar_chars

# Build Quoc Ngu to SinoNom dictionary from QuocNgu_SinoNom_Dic
quoc_ngu_to_sino_nom_dict = {}
for _, row in quoc_ngu_sino_nom_df.iterrows():
    quoc_ngu_char = row['QuocNgu']
    sino_nom_char = row['SinoNom']
    if quoc_ngu_char not in quoc_ngu_to_sino_nom_dict:
        quoc_ngu_to_sino_nom_dict[quoc_ngu_char] = set()
    quoc_ngu_to_sino_nom_dict[quoc_ngu_char].add(sino_nom_char)

def clean_data(sentence):
    # Remove punctuation
    sentence = re.sub(r'[.,;:!?"]', '', sentence)
    # Normalize text to lowercase
    sentence = sentence.lower()
    # Strip leading and trailing whitespace
    sentence = sentence.strip()
    # Add cleaned sentence to the list
    return sentence

def group_boxes_in_columns(bounding_boxes):
    grouped_columns = []  # List to store grouped columns
    current_group = []  # Temporary group for the current column

    for i, bbox in enumerate(bounding_boxes):
        if i == 0:
            # First box, start a new group
            current_group.append(bbox)
        else:
            # Compare the current box's left x-coordinate with the previous group's right x-coordinate
            current_left_x = bbox[0][1][0]
            previous_right_x = bounding_boxes[i - 1][0][0][0]

            if current_left_x < previous_right_x:
                # Not the same column, finalize the current group
                grouped_columns.append(current_group)
                # Start a new group
                current_group = [bbox]
            else:
                # Add to the current group
                current_group.append(bbox)

    # Add the last group if it exists
    if current_group:
        grouped_columns.append(current_group)

    return grouped_columns

def custom_comparator(item_a, item_b):
    a_bottom = item_a[0][2][1]
    b_bottom = item_b[0][2][1]
    a_top = item_a[0][1][1]
    b_top = item_b[0][1][1]
    a_right = item_a[0][1][0]
    b_right = item_b[0][1][0]
    a_left = item_a[0][0][0]
    b_left = item_b[0][0][0]

    if a_left + 1 >= b_right:
        return -1
    elif b_left >= a_right + 1:
        return 1
    else:
        if a_top < b_top:
            return -1
        else:
            return 1

def rearrange_with_custom_comparator(data):
    return sorted(data, key=cmp_to_key(custom_comparator))

def filter_bounding_boxes(bounding_boxes):
    """
    Filters out bounding boxes that lie outside the specified boundaries.

    Args:
        bounding_boxes (list of tuples): List of bounding boxes, where each box is represented as 
                                         (x_min, y_min, x_max, y_max).
        top (float): Top boundary.
        bottom (float): Bottom boundary.
        left (float): Left boundary.
        right (float): Right boundary.

    Returns:
        list of tuples: Filtered list of bounding boxes within the boundaries.
    """
    invalid_boxes = set()
    for i in range(0, len(bounding_boxes)):
        left = bounding_boxes[i][0][0][0]
        right = bounding_boxes[i][0][1][0]
        top = bounding_boxes[i][0][1][1]
        bottom = bounding_boxes[i][0][2][1]
        # Check if the box is within the boundaries
        if left < LEFT or right > RIGHT or top < TOP or bottom > BOTTOM:
            invalid_boxes.add(i)
    return invalid_boxes

def compute_cost(ocr_char, quoc_ngu_word):
    quoc_ngu_word = re.sub(r'[.,;:!?”“"]', '', quoc_ngu_word).lower()
    S1 = sino_nom_similar_dict.get(ocr_char, {ocr_char})
    S2 = quoc_ngu_to_sino_nom_dict.get(quoc_ngu_word, set())
    
    if ocr_char in S2:
        return 0, ocr_char

    intersection = [char for char in S1 if char in S2]
    
    if len(intersection) >= 1:
        return 0.5, list(intersection)[0]
    else:
        return 2, None # Levenstein: substitution cost = 2

def med_with_custom_cost(sino_nom_string, quoc_ngu_string):
    # quoc_ngu_string = clean_data(quoc_ngu_string)
    quoc_ngu_string = quoc_ngu_string.split()
    m, n = len(sino_nom_string), len(quoc_ngu_string)
    dp = np.zeros((m + 1, n + 1))
    
    # Initialize dp array for insertion/deletion costs
    for i in range(1, m + 1):
        dp[i][0] = i
    for j in range(1, n + 1):
        dp[0][j] = j

    aligned_result = []

    # Fill dp table with computed costs
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost, _ = compute_cost(sino_nom_string[i - 1], quoc_ngu_string[j - 1])
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    # Backtrack to find the alignment path
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost, aligned_char = compute_cost(sino_nom_string[i - 1], quoc_ngu_string[j - 1])
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                status = "match" if cost == 0 else "partial match" if cost == 0.5 else "not match"
                aligned_result.append((sino_nom_string[i - 1], status))
                i, j = i - 1, j - 1
                continue
        if i > 0 and (j == 0 or dp[i][j] == dp[i - 1][j] + 1):
            aligned_result.append((sino_nom_string[i - 1], "not match"))
            i -= 1
        elif j > 0:
            aligned_result.append(('-', "not match"))
            j -= 1

    aligned_result.reverse()
    return aligned_result, dp[m][n]

# def character_align(columns, quoc_ngu_sentences, invalid_boxes, i):
#     alignments = []
#     box_index = 0
#     sentence_index = 0

#     # Align columns and sentences, handling cases where not all columns have sentences
#     for column in columns:
#         boxes = [box for box in column]
#         valid_boxes = [box for idx, box in enumerate(column) if idx + box_index not in invalid_boxes]
#         box_index += len(column)  # Update box index for next column
        
#         # Check if there are still sentences to align
#         if sentence_index < len(quoc_ngu_sentences):
#             if len(valid_boxes) > 0:
#                 alignments.append((valid_boxes, quoc_ngu_sentences[sentence_index]))
#                 sentence_index += 1
#             else:
#                 alignments.append((boxes, None))
#         else:
#             # No sentence available for this column
#             alignments.append((valid_boxes, None))

#     # Write results to Excel
#     workbook = xlsxwriter.Workbook('output.xlsx')
#     worksheet = workbook.add_worksheet("Alignment Output")

#     # Define formatting
#     font_perfect_match = workbook.add_format({'font_name': "Nom Na Tong", 'font_size': 14, 'color': 'black'})
#     font_partial_match = workbook.add_format({'font_name': "Nom Na Tong", 'font_size': 14, 'color': 'blue'})
#     font_no_match = workbook.add_format({'font_name': "Nom Na Tong", 'font_size': 14, 'color': 'red'})
#     font_false_box = workbook.add_format({'font_name': "Nom Na Tong", 'font_size': 14, 'color': 'green'})

#     # Headers
#     worksheet.write_row(0, 0, ["ID", "Image Box", "SinoNom OCR", "Chữ Quốc Ngữ"])

#     # Process alignments
#     for index, (valid_boxes, quoc_ngu_sentence) in enumerate(alignments, start=1):
#         # Generate unique box ID
#         box_id = f"ppp{i}_ss{index}"

#         # Construct SinoNom OCR string from valid boxes
#         sino_nom_string = "".join(box[1][0] for box in valid_boxes) if valid_boxes else None

#         # Perform alignment if both SinoNom OCR and Quốc Ngữ sentence exist
#         if sino_nom_string and quoc_ngu_sentence:
#             aligned_result, _ = med_with_custom_cost(sino_nom_string, quoc_ngu_sentence)
#         else:
#             aligned_result = []

#         # Format SinoNom OCR output
#         sino_nom_output = []
#         for sino_char, status in aligned_result:
#             if status == "match":
#                 sino_nom_output.extend([font_perfect_match, sino_char])
#             elif status == "partial match":
#                 sino_nom_output.extend([font_partial_match, sino_char])
#             else:
#                 sino_nom_output.extend([font_no_match, sino_char])

#         # Write data to Excel
#         worksheet.write(index, 0, box_id)
#         # Image boxes in new lines
#         worksheet.write(index, 1, str([box[0] for box in valid_boxes]) if valid_boxes else "Invalid", font_false_box if not valid_boxes else None)
        
#         if sino_nom_output:
#             worksheet.write_rich_string(index, 2, *sino_nom_output)
#         else:
#             worksheet.write(index, 2, sino_nom_string or "No OCR", font_false_box)
        
#         worksheet.write(index, 3, quoc_ngu_sentence or "No Sentence")

#     workbook.close()

# def process_single_box_text(box_path, text_path, i):
#     with open(box_path, 'r', encoding='utf-8') as file:
#         data = json.load(file)
#         bounding_boxes = data.get("result_bbox", [])
#         bounding_boxes = rearrange_with_custom_comparator(bounding_boxes)
#         columns = group_boxes_in_columns(bounding_boxes)
#         invalid_boxes = filter_bounding_boxes(bounding_boxes)

#     with open(text_path, 'r', encoding='utf-8') as file:
#         quoc_ngu_sentences = json.load(file)

#     box_index = 0
#     for column in columns:
#         if len(column) == 1 and calculate_bbox_length(column[0][0]) <= 21:
#             invalid_boxes.add(box_index)
#             box_index += 1
#         else:
#             box_index += len(column)

#     character_align(columns, quoc_ngu_sentences, invalid_boxes, i)

def calculate_bbox_length(bbox):
    """
    Calculate the length of a bounding box based on its coordinates.
    
    Args:
        bbox (list): A bounding box represented as a list of four points (x, y).
    
    Returns:
        float: The height of the bounding box.
    """
    # Calculate the vertical length (difference between bottom and top y-coordinates)
    top_y = bbox[0][1]  # y-coordinate of the top-left corner
    bottom_y = bbox[2][1]  # y-coordinate of the bottom-right corner
    return abs(bottom_y - top_y)
def process_single_box_text(box_path, text_path, index, worksheet, current_row):
    # Check if the text file exists
    if not os.path.exists(text_path):
        print(f"Warning: {text_path} does not exist.")
        quoc_ngu_sentences = []  # Default empty list if the file is missing
    else:
        try:
            # Open and read the text file
            with open(text_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
                # Check if file is empty
                if not content:
                    print(f"Warning: {text_path} is empty.")
                    quoc_ngu_sentences = []  # Default empty list if file is empty
                else:
                    try:
                        # Try loading the content as JSON
                        quoc_ngu_sentences = json.loads(content)
                    except json.JSONDecodeError:
                        print(f"Error: {text_path} contains invalid JSON.")
                        quoc_ngu_sentences = []  # Default empty list if JSON is invalid

        except Exception as e:
            print(f"Error reading file {text_path}: {e}")
            quoc_ngu_sentences = []  # Default empty list on other errors

    # Processing the sentences from the JSON file (assuming it's a list of sentences or text blocks)
    for sentence in quoc_ngu_sentences:
        # Example of processing the sentences (you can replace with your actual processing logic)
        print(f"Processing sentence {index}: {sentence}")
        
        # Assuming `worksheet` is a list and `current_row` is an index to place the data in the worksheet
        worksheet[current_row] = sentence
        current_row += 1  # Move to the next row for the next sentence

    # Returning updated row index for the next process
    return current_row


def character_align(columns, quoc_ngu_sentences, invalid_boxes, i, worksheet, start_row):
    alignments = []
    box_index = 0
    sentence_index = 0

    # Align columns and sentences, handling cases where not all columns have sentences
    for column in columns:
        boxes = [box for box in column]
        valid_boxes = [box for idx, box in enumerate(column) if idx + box_index not in invalid_boxes]
        box_index += len(column)  # Update box index for next column
        
        # Check if there are still sentences to align
        if sentence_index < len(quoc_ngu_sentences):
            if len(valid_boxes) > 0:
                alignments.append((valid_boxes, quoc_ngu_sentences[sentence_index]))
                sentence_index += 1
            else:
                alignments.append((boxes, None))
        else:
            # No sentence available for this column
            alignments.append((boxes, None))

    # Process alignments and append to worksheet
    current_row = start_row
    font_perfect_match = workbook.add_format({'font_name': "Nom Na Tong", 'font_size': 14, 'color': 'black'})
    font_partial_match = workbook.add_format({'font_name': "Nom Na Tong", 'font_size': 14, 'color': 'blue'})
    font_no_match = workbook.add_format({'font_name': "Nom Na Tong", 'font_size': 14, 'color': 'red'})
    font_false_box = workbook.add_format({'font_name': "Nom Na Tong", 'font_size': 14, 'color': 'green'})

    for index, (valid_boxes, quoc_ngu_sentence) in enumerate(alignments, start=1):
        # Generate unique box ID
        box_id = f"ppp{i}_ss{index}"

        # Construct SinoNom OCR string from valid boxes
        sino_nom_string = "".join(box[1][0] for box in valid_boxes) if valid_boxes else None

        # Perform alignment if both SinoNom OCR and Quốc Ngữ sentence exist
        if sino_nom_string and quoc_ngu_sentence:
            aligned_result, _ = med_with_custom_cost(sino_nom_string, quoc_ngu_sentence)
        else:
            aligned_result = []

        # Format SinoNom OCR output
        sino_nom_output = []
        for sino_char, status in aligned_result:
            if status == "match":
                sino_nom_output.extend([font_perfect_match, sino_char])
            elif status == "partial match":
                sino_nom_output.extend([font_partial_match, sino_char])
            else:
                sino_nom_output.extend([font_no_match, sino_char])

        # Write data to worksheet
        worksheet.write(current_row, 0, box_id)
        # Image boxes in new lines
        worksheet.write(current_row, 1, str([box[0] for box in valid_boxes]) if valid_boxes else "Invalid", font_false_box if not valid_boxes else None)
        
        if sino_nom_output:
            worksheet.write_rich_string(current_row, 2, *sino_nom_output)
        else:
            worksheet.write(current_row, 2, sino_nom_string or "No OCR", font_false_box)
        
        worksheet.write(current_row, 3, quoc_ngu_sentence or "No Sentence")
        current_row += 1  # Move to the next row

    return current_row

# Main loop to process all pairs of files
workbook = xlsxwriter.Workbook('output.xlsx')  # Create workbook once
worksheet = workbook.add_worksheet("Alignment Output")

# Add headers
worksheet.write_row(0, 0, ["ID", "Image Box", "SinoNom OCR", "Chữ Quốc Ngữ"])
current_row = 1  # Start writing data below the headers

for i in range(6, 67, 2):
    box_path = f"{BOX_PATH_PREFIX}{i}{BOX_PATH_SUFFIX}"
    text_path = f"{TEXT_PATH_PREFIX}{i+1}{TEXT_PATH_SUFFIX}"
    current_row = process_single_box_text(box_path, text_path, i, worksheet, current_row)

workbook.close()  # Save the workbook
