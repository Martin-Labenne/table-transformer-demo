import sys
import os
from pathlib import Path

# Give priority to the modules that are inside tableTransformer 
# and handles the conflict between tableTransformer/detr/datasets and hugging face datasets' library
sys.path.insert(1, os.getcwd() + '/tableTransformer/src')
sys.path.insert(2, os.getcwd() + '/tableTransformer/detr')
# This snipet is necessary for inference to work

from tableTransformer.src.inference import TableExtractionPipeline, rescale_bboxes, objects_to_structures, \
    structure_to_cells, cells_to_html, cells_to_csv, objects_to_crops, output_result

# Image to Tensor pipelines
from tableTransformer.src.inference import structure_transform, detection_transform


def _outputs_to_objects(outputs, img_size, class_idx2name):
    m = outputs.logits.softmax(-1).max(-1)
    pred_labels = list(m.indices.detach().cpu().numpy())[0]
    pred_scores = list(m.values.detach().cpu().numpy())[0]
    pred_bboxes = outputs['pred_boxes'].detach().cpu()[0]
    pred_bboxes = [elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)]

    objects = []
    for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
        class_label = class_idx2name[int(label)]
        if not class_label == 'no object':
            objects.append({'label': class_label, 'score': float(score),
                            'bbox': [float(elem) for elem in bbox]})

    return objects

def log_extracted_tables(
        extracted_tables,
        image_filename,
        log_dir,
        vizualise=False,
        verbose=False
    ):
    # Create log folder if not exists
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    args = type('Args', (object,), {
        'visualize': vizualise, 
        'out_dir': log_dir,
        'verbose': verbose
    } )

    image_file_stem, image_file_ext = os.path.splitext(image_filename)

    for table_idx, extracted_table in enumerate(extracted_tables): 
        file_suffix = f'_crop_{table_idx}' if table_idx is not None else '_crop'

        for key, val in extracted_table.items():
            output_result(
                key, val, args, 
                extracted_table['image'], 
                f'{image_file_stem}{file_suffix}{image_file_ext}'
            )

class PretrainTableExtractionPipeline(TableExtractionPipeline): 
    def __init__(
        self, 
        det_device=None, str_device=None,
        det_model=None, str_model=None
    ):
        super().__init__(
            det_device=det_device, str_device=str_device
        )
        if det_model is not None: 
            self.det_model = det_model
            self.det_model.to(det_device)
        if str_model is not None: 
            self.str_model = str_model
            self.str_model.to(str_device)


    """
    `tokens` is expected to be a list of dictionaries containing a list of words and their bounding boxes in image coordinates. 
    It is assumed to be sorted in reading order. The format for tokens is:  
    ```python
    [
        {
            'bbox': [0.0, 0.0, 50.0, 50.0],
            'text': 'First'
            'span_num': 1,
            'line_num': <<line_num>> || 0,
            'block_num': <<block_num>> || 0
        },
        {
            'bbox': [52.0, 0.0, 102.0, 50.0],
            'text': 'next'
            'span_num': 2,
            'line_num': <<line_num>> || 0,
            'block_num': <<block_num>> || 0
        }
    ]
    ```
    """

    def detect(
        self, 
        img, tokens=[], 
        out_objects=True, 
        out_crops=True, 
        crop_padding=10
    ):
        out_formats = {}
        if self.det_model is None:
            print("No detection model loaded.")
            return out_formats

        # Transform the image how the model expects it
        img_tensor = detection_transform(img)

        # Run input image through the model
        outputs = self.det_model(img_tensor.unsqueeze(0).to(self.det_device))

        # Post-process detected objects, assign class labels
        objects = _outputs_to_objects(outputs, img.size, self.det_class_idx2name)
        if out_objects:
            out_formats['objects'] = objects
        if not out_crops:
            return out_formats

        # Crop image and tokens for detected table
        if out_crops:
            tables_crops = objects_to_crops(
                img, tokens, 
                objects, 
                self.det_class_thresholds,
                padding=crop_padding
            )

            out_formats['crops'] = tables_crops
        
        return out_formats

    def recognize(
        self, 
        img, tokens=[], 
        out_objects=False, 
        out_cells=False,
        out_html=False, 
        out_csv=False
    ):
        out_formats = {}
        if self.str_model is None:
            print("No structure model loaded.")
            return out_formats

        if not (out_objects or out_cells or out_html or out_csv):
            print("No output format specified")
            return out_formats

        # Transform the image how the model expects it
        img_tensor = structure_transform(img)

        # Run input image through the model
        outputs = self.str_model(img_tensor.unsqueeze(0).to(self.str_device))

        # Post-process detected objects, assign class labels
        objects = _outputs_to_objects(outputs, img.size, self.str_class_idx2name)
        if out_objects:
            out_formats['objects'] = objects
        if not (out_cells or out_html or out_csv):
            return out_formats

        # Further process the detected objects so they correspond to a consistent table 
        tables_structure = objects_to_structures(objects, tokens, self.str_class_thresholds)

        # Enumerate all table cells: grid cells and spanning cells
        tables_cells = [structure_to_cells(structure, tokens)[0] for structure in tables_structure]
        if out_cells:
            out_formats['cells'] = tables_cells

        # Convert cells to HTML
        if out_html:
            tables_htmls = [cells_to_html(cells) for cells in tables_cells]
            out_formats['html'] = tables_htmls

        # Convert cells to CSV, including flattening multi-row column headers to a single row 
        if out_csv:
            tables_csvs = [cells_to_csv(cells) for cells in tables_cells]
            out_formats['csv'] = tables_csvs

        return out_formats