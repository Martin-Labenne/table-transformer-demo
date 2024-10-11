from tableExtraction import PretrainTableExtractionPipeline, log_extracted_tables
from tokenExtraction import TokenReader

from transformers import TableTransformerForObjectDetection, AutoModelForObjectDetection
from PIL import Image
from numpy import array

TABLE_DETECTION_MODEL_PATH = "microsoft/table-transformer-detection"
TABLE_STRUCTURE_MODEL_PATH = "microsoft/table-transformer-structure-recognition-v1.1-all"

DEFAULT_DEVICE = 'cpu'
DEFAULT_OCR_LANGUAGE = 'en'

DEFAULT_OUT_OPTIONS = {
    'out_objects': True,
    'out_cells': True,
    'out_html': True,
    'out_csv': True
}

class DocumentTableProcessor(object): 
    OCR_EARLY = 'early'
    OCR_MID = 'mid'
    OCR_LATE = 'late'
    OCR_STRATEGIES = [OCR_EARLY, OCR_MID, OCR_LATE]

    table_detection_model = AutoModelForObjectDetection.from_pretrained(TABLE_DETECTION_MODEL_PATH, revision="no_timm")
    table_structure_model = TableTransformerForObjectDetection.from_pretrained(TABLE_STRUCTURE_MODEL_PATH)

    def __init__(
        self, 
        det_device=DEFAULT_DEVICE, 
        str_device=DEFAULT_DEVICE,
        ocr_device=DEFAULT_DEVICE, 
        ocr_language=DEFAULT_OCR_LANGUAGE,
        ocr_strategy=OCR_EARLY
    ) :
        
        self.det_device = det_device
        self.str_device = str_device
        self.ocr_device = ocr_device
        self.ocr_language = ocr_language

        if ocr_strategy not in self.OCR_STRATEGIES:
            raise ValueError(f"Invalid ocr_strategy. Expected one of {self.OCR_STRATEGIES}")
        else : 
            self.ocr_strategy = ocr_strategy

        self.token_reader = TokenReader(
            language=self.ocr_language, 
            device=self.ocr_device
        )

        self.table_extraction_pipeline = PretrainTableExtractionPipeline(
            det_device=self.det_device,
            str_device=self.str_device, 
            det_model=self.table_detection_model,
            str_model=self.table_structure_model
        )

    def _extract_strat_early(
        self, 
        image: Image.Image, 
        readtext_args: dict = None, 
        out_options: dict = None
    ): 
        tokens = self.token_reader.get_tokens(
            array(image.convert('L')), readtext_args
        )
        extracted_tables = self.table_extraction_pipeline.extract(
            image.convert('RGB'), tokens=tokens, **out_options
        )
        return extracted_tables

    def _extract_strat_mid(
        self, 
        image: Image.Image,
        readtext_args: dict = None, 
        out_options: dict = None
    ):
        detected_tables = self.table_extraction_pipeline.detect(image.convert('RGB'))

        extracted_tables = []
        for crop_table in detected_tables['crops']: 
            crop_image = crop_table['image']
            crop_tokens = self.token_reader.get_tokens(
                array(crop_image.convert('L')), readtext_args
            )

            extracted_table = self.table_extraction_pipeline.recognize(
                crop_image, crop_tokens, **out_options
            )
            extracted_table['image'] = crop_image
            extracted_table['tokens'] = crop_tokens
            extracted_tables.append(extracted_table)

        return extracted_tables

    def extract(
        self, 
        image_path: str, 
        readtext_args: dict = None, 
        out_options: dict = None,
        log: bool = False, 
        log_options: dict = None
    ): 
        
        image = Image.open(image_path)

        if self.ocr_strategy == self.OCR_EARLY : 

            extracted_tables = self._extract_strat_early(
                image, readtext_args,
                out_options if out_options is not None else self.DEFAULT_OUT_OPTIONS
            )

        elif self.ocr_strategy == self.OCR_MID :

            extracted_tables = self._extract_strat_mid(
                image, readtext_args,
                out_options if out_options is not None else self.DEFAULT_OUT_OPTIONS
            )

        elif self.ocr_strategy == self.OCR_LATE: 
            raise NotImplementedError(f'The {self.OCR_LATE} strategy is not implemented yet.')

        if log: 
            log_extracted_tables(extracted_tables, image_path, **log_options)

        return extracted_tables