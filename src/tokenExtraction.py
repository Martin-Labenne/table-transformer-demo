from easyocr import Reader
import fitz

class TokenReader(Reader): 
    def __init__(self, language, device):
        gpu = device.lower() == "gpu" 
        super().__init__(
            lang_list=[language], 
            gpu=gpu
        )

    def get_tokens(self, image, readtext_args):
        reader_results = self.readtext(image, **readtext_args)
        tokens = [
            {
                'bbox': [
                    float(coord) for coord in [ 
                        bbox_coord[0][0], 
                        bbox_coord[0][1], 
                        bbox_coord[2][0], 
                        bbox_coord[2][1] 
                    ]
                ],
                'text': text,
                'proba': proba,
                'block_num': 0, 
                'line_num': 0, 
                'span_num': i,
            }
            for i, (bbox_coord, text, proba) in enumerate(reader_results)
        ] 

        return tokens

class PDFTokenReader(): 
    def __init__(self, dpi=100):
        self.flags = fitz.TEXT_INHIBIT_SPACES & ~fitz.TEXT_PRESERVE_IMAGES
        self.default_pdf_dpi = 72
        self.dpi = dpi

    def get_tokens(self, page): 

        # Thanks https://github.com/microsoft/table-transformer/issues/121    

        words = page.get_text(option="words", flags=self.flags)
        # make sure the bounding boxes are in the same scale as the generated image
        dpi_scale = self.dpi/self.default_pdf_dpi
        tokens = []
        for word_meta in words:
            pdf_scaled_word_bbox = [ word_meta[0], word_meta[1], word_meta[2], word_meta[3] ]
            tokens.append({
                'bbox': [ 
                    word_cord * dpi_scale 
                    for word_cord in pdf_scaled_word_bbox
                ], 
                'text': word_meta[4],
                'proba': 1, 
                # 'flags': 0,
                'block_num': word_meta[5],
                'line_num': word_meta[6],
                'span_num': word_meta[7]
            })    
        
        return tokens