import json
import xlsxwriter
from collections import defaultdict
import pandas as pd
import ast
import numpy as np
from extract_phien_am import *
quoc_ngu_df = pd.read_excel('QuocNgu_SinoNom_Dic.xlsx')
sino_nom_df = pd.read_excel('SinoNom_similar_Dic.xlsx')
sino_nom_df['Top 20 Similar Characters'] = sino_nom_df['Top 20 Similar Characters'].apply(ast.literal_eval)
def get_all_sino_nom_from_quoc_ngu(df, quoc_ngu):
    quoc_ngu = quoc_ngu.lower()
    return df[df['QuocNgu'] == quoc_ngu]['SinoNom'].tolist()
def get_similar_sino_nom_from_sino_nom(df, sino_nom):
    sino_nom = sino_nom.lower()
    values = np.array(df[df['Input Character'] == sino_nom]['Top 20 Similar Characters'].values.tolist()).reshape(-1)
    return values
def get_intersection(sino_nom, quoc_ngu):
    set1 = set(get_similar_sino_nom_from_sino_nom(sino_nom_df, sino_nom))
    set1.add(sino_nom)
    set2 = set(get_all_sino_nom_from_quoc_ngu(quoc_ngu_df, quoc_ngu))
    return set1.intersection(set2)
def is_match(sino_nom, quoc_ngu):
    return len(get_intersection(sino_nom, quoc_ngu)) > 0
def align_strings(sino_nom_string, quoc_ngu_string):
    quoc_ngu_string = quoc_ngu_string.split()
    m, n = len(sino_nom_string), len(quoc_ngu_string)
    dp = np.zeros((m + 1, n + 1))
    
    for i in range(1, m + 1):
        dp[i][0] = i
    for j in range(1, n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if is_match(sino_nom_string[i - 1], quoc_ngu_string[j - 1]):
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
    aligned_sino_nom, aligned_quoc_ngu = [], []             
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and is_match(sino_nom_string[i - 1], quoc_ngu_string[j - 1]):
            aligned_sino_nom.append(sino_nom_string[i - 1])
            aligned_quoc_ngu.append(quoc_ngu_string[j - 1])
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or dp[i][j] == dp[i - 1][j] + 1):
            aligned_sino_nom.append(sino_nom_string[i - 1])
            aligned_quoc_ngu.append('-')
            i -= 1
        elif j > 0 and (i == 0 or dp[i][j] == dp[i][j - 1] + 1):
            aligned_sino_nom.append('-')
            aligned_quoc_ngu.append(quoc_ngu_string[j - 1])
            j -= 1
        else:
            aligned_sino_nom.append(sino_nom_string[i - 1])
            aligned_quoc_ngu.append(quoc_ngu_string[j - 1])
            i -= 1
            j -= 1

    aligned_sino_nom.reverse()
    aligned_quoc_ngu.reverse()
    return ''.join(aligned_sino_nom), ' '.join(aligned_quoc_ngu)

def read_response_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            image_name = content[:content.index('[')]
            image_name = image_name.strip()
        try:
            start_index = content.index('[')
            json_data = json.loads(content[start_index:])
            return image_name, json_data
        except Exception as e:
            print("Error parsing JSON from text file:", e)
            exit()
    except UnicodeDecodeError as e:
        print(f"Encoding error: {e}")
        return None


image_name, json_data = read_response_file('response/thanh_giao_yeu_ly_image_3.txt')
output_array = [
    [item["text"], item["confidence"], item["points"]]
    for item in json_data
]

