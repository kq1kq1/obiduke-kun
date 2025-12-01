import os
import cv2
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from PyPDF2 import PdfReader, PdfWriter
import tempfile
import io


class PDFProcessor:
    def __init__(self):
        self.competitor_banners_path = 'uploads/banners/competitor'
        self.custom_banners_path = 'uploads/banners/custom'
        self.match_threshold = 0.8
    
    def load_competitor_templates(self):
        templates = []
        if os.path.exists(self.competitor_banners_path):
            for filename in os.listdir(self.competitor_banners_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    template_path = os.path.join(self.competitor_banners_path, filename)
                    template = cv2.imread(template_path)
                    if template is not None:
                        templates.append(template)
        return templates
    
    def get_custom_banner(self):
        if os.path.exists(self.custom_banners_path):
            for filename in os.listdir(self.custom_banners_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    return os.path.join(self.custom_banners_path, filename)
        return None
    
    def detect_and_remove_banner(self, image, templates):
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        mask = np.zeros(img_cv.shape[:2], dtype=np.uint8)
        
        for template in templates:
            if template.shape[0] > img_cv.shape[0] or template.shape[1] > img_cv.shape[1]:
                continue
            
            result = cv2.matchTemplate(img_cv, template, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= self.match_threshold)
            
            for pt in zip(*locations[::-1]):
                h, w = template.shape[:2]
                mask[pt[1]:pt[1]+h, pt[0]:pt[0]+w] = 255
        
        if np.any(mask):
            img_cv = cv2.inpaint(img_cv, mask, 3, cv2.INPAINT_TELEA)
        
        return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    
    def add_custom_banner(self, image, banner_path):
        banner = Image.open(banner_path)
        
        img_width, img_height = image.size
        banner_width, banner_height = banner.size
        
        scale = img_width / banner_width
        new_banner_height = int(banner_height * scale)
        banner = banner.resize((img_width, new_banner_height), Image.Resampling.LANCZOS)
        
        new_height = img_height + new_banner_height
        new_image = Image.new('RGB', (img_width, new_height), (255, 255, 255))
        new_image.paste(image, (0, 0))
        
        if banner.mode == 'RGBA':
            new_image.paste(banner, (0, img_height), banner)
        else:
            new_image.paste(banner, (0, img_height))
        
        return new_image
    
    def process_pdf(self, input_path, output_path):
        templates = self.load_competitor_templates()
        custom_banner = self.get_custom_banner()
        
        images = convert_from_path(input_path, dpi=150)
        processed_images = []
        
        for image in images:
            processed = image
            
            if templates:
                processed = self.detect_and_remove_banner(processed, templates)
            
            if custom_banner:
                processed = self.add_custom_banner(processed, custom_banner)
            
            processed_images.append(processed)
        
        if processed_images:
            first_image = processed_images[0]
            if len(processed_images) > 1:
                first_image.save(
                    output_path,
                    'PDF',
                    save_all=True,
                    append_images=processed_images[1:]
                )
            else:
                first_image.save(output_path, 'PDF')
        
        return output_path
