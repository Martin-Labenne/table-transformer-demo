from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
import pandas as pd
import io
import os
import fitz

from src.documentTableProcessor import DocumentTableProcessor

from dotenv import load_dotenv
load_dotenv() 


# Initialize FastAPI app
app = FastAPI()

# Assuming DocumentTableProcessor is already defined and imported
document_table_processor = DocumentTableProcessor(
    det_device=os.environ['TABLE_DETECTION_DEVICE'],
    str_device=os.environ['TABLE_STRUCTURE_DEVICE'], 
    ocr_device=os.environ['READER_DEVICE'],
    ocr_language=os.environ['READER_LANGUAGE'],
    ocr_strategy='early'
)

@app.post("/extract-table")
async def extract_table(file: UploadFile = File(...)):
    try:

        file_name, file_extention = os.path.splitext(file.filename)
        print(file_extention)

        excel_filename = f'{ file_name }_extracted_tables.xlsx'

        if file_extention == '.pdf':
            pdf = fitz.open(stream=io.BytesIO(await file.read()), filetype='pdf')
            extracted_tables = document_table_processor.extract_pdf(pdf)

        else: 
            # Read the uploaded file as an image
            image = Image.open(io.BytesIO(await file.read()))

            # easyocr args
            readtext_args = {'low_text': 0.3}  

            # Process the image and extract tables
            extracted_tables = document_table_processor.extract(
                image=image,
                readtext_args=readtext_args
            )
        
        # Init an in memory buffer
        excel_io = io.BytesIO()
        
        with pd.ExcelWriter(excel_io, engine='openpyxl') as writer:

            for i, table in enumerate(extracted_tables):

                csv_content = table['csv'][0]
                df = pd.read_csv( io.StringIO(csv_content) )

                 # Write each DataFrame to a separate sheet
                sheet_name = f'Sheet{ i + 1 }'
                df.to_excel(writer, index=False, sheet_name=sheet_name)

        # Reset pointer to the start of the file
        # Ensures that the StreamingResponse starts reading the data from the beginning of the file
        excel_io.seek(0)  

        # Return Excel file as a streaming response
        response = StreamingResponse(
            excel_io,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        response.headers["Content-Disposition"] = f"attachment; filename={ excel_filename }"
        return response

    except Exception as e:
        return {"error": str(e)}
