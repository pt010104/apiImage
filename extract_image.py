import fitz
import os
output_images_dir = "extracted_images"
os.makedirs(output_images_dir, exist_ok=True)

def extract_images_from_pdf(pdf_path, phien_am_pages):
    file_name = os.path.splitext(os.path.basename(pdf_path))[0]
    pdf_document = fitz.open(pdf_path)
    image_filenames = []
    for page_number in range(pdf_document.page_count):
        if page_number + 1 not in phien_am_pages:
            continue
        page = pdf_document[page_number]
        image_list = page.get_images(full=True)
        for image_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_filename = f"{file_name}_image_{str(page_number)}.{image_ext}"
            image_filenames.append(image_filename)
            image_path = os.path.join(output_images_dir, image_filename)
            with open(image_path, "wb") as img_file:
                img_file.write(image_bytes)   