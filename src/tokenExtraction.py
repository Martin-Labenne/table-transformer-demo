from easyocr import Reader

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
                'span_num': i,
                'line_num': 0,
                'block_num': 0
            }
            for i, (bbox_coord, text, proba) in enumerate(reader_results)
        ] 

        return tokens

