# QSMnet+ API

## Overview
This API provides a Flask-based REST endpoint for making inferences using the QSMnet+ model.

## API Usage

### Endpoint
`POST /predict`

### Input Formats
The API accepts inputs in two formats:

1. **JSON Format**
   - Content-Type: `application/json`
   - Body structure:
     ```json
     {
       "volume": [[[float_values]]]  // 3D array representing the volume
     }
     ```

2. **MAT File Upload**
   - Content-Type: `multipart/form-data`
   - Form field: `file`
   - File type: MATLAB `.mat` file containing a 3D volume

### Response Format
The API returns a JSON response with the following structure:
