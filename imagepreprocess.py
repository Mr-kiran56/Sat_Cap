from PIL import Image, ImageFilter, ImageEnhance

class Preprocess:
    def __init__(self, img_path):
        self.img_path = img_path

    def crop_salient_region(self):
        try:
            img = Image.open(self.img_path).convert("RGB")

            gray = img.convert("L")
            gray = ImageEnhance.Contrast(gray).enhance(2.0)
            edges = gray.filter(ImageFilter.FIND_EDGES)

            bbox = edges.getbbox()
            if bbox:
                img = img.crop(bbox)

            img = img.resize((224, 224))

            return img
        
        except Exception as e:
            print(f" Error processing {self.img_path}: {e}")
            return Image.new('RGB', (224, 224), color='gray')
