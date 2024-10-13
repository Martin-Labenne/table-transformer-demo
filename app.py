from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
import pandas as pd
import io
import os

from documentTableProcessor import DocumentTableProcessor

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
        # Read the uploaded file as an image
        image = Image.open(io.BytesIO(await file.read()))

        # Set readtext arguments and output options
        readtext_args = {'low_text': 0.3}  # Add other args if needed
        out_options = {
            'out_objects': True,
            'out_cells': True,
            'out_html': True,
            'out_csv': True
        }

        # Process the image and extract tables
        extracted_tables = document_table_processor.extract(
            image=image,
            readtext_args=readtext_args,
            out_options=out_options
        )

         # Get the first table as a DataFrame (assuming CSV content can be loaded into DataFrame)
        csv_content = extracted_tables[0]['csv'][0]
        df = pd.read_csv(io.StringIO(csv_content))

        # Save the DataFrame as an Excel file in memory
        excel_io = io.BytesIO()
        with pd.ExcelWriter(excel_io, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
        excel_io.seek(0)  # Reset pointer to the start of the file

        # Return Excel file as a streaming response
        response = StreamingResponse(
            excel_io,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        response.headers["Content-Disposition"] = f"attachment; filename=extracted_table.xlsx"
        return response
    

    except Exception as e:
        return {"error": str(e)}

# To run the app: uvicorn your_script_name:app --reload
