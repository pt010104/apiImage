import fitz
import string
import re
def get_total_pages(pdf_document):
    return pdf_document.page_count

def is_phien_am_page(pdf_document, page_number, min_length_threshold=100):
    if page_number == 0:
        return False
    previous_page = pdf_document[page_number - 1]
    image_list = previous_page.get_images(full=True)
    if not image_list:
        return False
    current_page = pdf_document[page_number]
    text = current_page.get_text()
    return len(''.join(text.split("\n"))) > min_length_threshold
def get_phien_am_sentences(pdf_document, page_number):
    drop_parts = ['chú thích', 'phiên dịch', 'dịch nghĩa']
    special_pages = [4]
    page = pdf_document[page_number]
    text_data = page.get_text("dict")
    sorted_text = []
    for block in text_data.get("blocks", []):
        if "lines" not in block:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                bbox = span.get("bbox", None)
                content = span.get("text", "")
                if bbox and content:
                    sorted_text.append((bbox, content))
    sorted_text = sorted(sorted_text, key=lambda x: (x[0][1], x[0][0]))
    sorted_content = "".join([text[1] for text in sorted_text])
    raw_sentences = sorted_content.split("\uf022")
    raw_sentences = raw_sentences[1:]
    processed_sentences = []
    for sentence in raw_sentences:
        if "\n" in sentence:
            last_newline_index = sentence.rindex("\n")
            sentence = sentence[:last_newline_index]

        for part in drop_parts:
            if part in sentence.lower():
                part_index = sentence.lower().index(part)
                sentence = sentence[:part_index]
        if page_number in special_pages:
            sentence = re.sub(r"\s*\([^()]*\)\s*", " ", sentence)
            while re.search(r"\([^()]*\)", sentence):
                sentence = re.sub(r"\([^()]*\)", " ", sentence)
            sentence = re.sub(r"\s+", " ", sentence).strip()
        translation_table = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
        sentence = sentence.translate(translation_table)
        sentence = re.sub(r'\d+', '', sentence)
        sentence = sentence.replace("\n", " ").replace("\t", " ").replace("-", " ")
        sentence = " ".join(sentence.split())
        processed_sentences.append(sentence)
    return processed_sentences






