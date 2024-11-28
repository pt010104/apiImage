from extract_phien_am import *
from extract_image import *
from image_pre_process import *
pdf_path = "thanh_giao_yeu_ly.pdf"
output_txt_file = "phien_am_pages.txt"
phien_am_pages = []


pdf_document = fitz.open(pdf_path)
total_pages = get_total_pages(pdf_document)
for i in range(get_total_pages(pdf_document)):
    if is_phien_am_page(pdf_document, i):
        phien_am_pages.append(i)
extract_images_from_pdf(pdf_path, phien_am_pages)

pdf_document.close()
        
