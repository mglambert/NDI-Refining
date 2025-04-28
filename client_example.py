import requests
import numpy as np
import scipy.io
import argparse
import os
import json

def send_json_request(api_url, volume_data):
    """
    Send a 3D volume to the API as JSON data
    
    Args:
        api_url (str): The URL of the API endpoint
        volume_data (numpy.ndarray): 3D volume data
        
    Returns:
        dict: API response
    """
    print(f"Sending JSON request to {api_url}...")
    print(f"Volume shape: {volume_data.shape}")
    
    # Convert numpy array to list for JSON serialization
    volume_list = volume_data.tolist()
    
    # Prepare the payload
    payload = {
        'volume': volume_list
    }
    
    # Send the request
    response = requests.post(
        api_url,
        json=payload,
        headers={'Content-Type': 'application/json'}
    )
    
    # Process the response
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

def send_mat_file_request(api_url, mat_file_path):
    """
    Send a .mat file to the API
    
    Args:
        api_url (str): The URL of the API endpoint
        mat_file_path (str): Path to the .mat file
        
    Returns:
        dict: API response
    """
    print(f"Sending .mat file request to {api_url}...")
    print(f"File path: {mat_file_path}")
    
    # Check if file exists
    if not os.path.exists(mat_file_path):
        print(f"Error: File {mat_file_path} not found")
        return None
    
    # Prepare the file upload
    with open(mat_file_path, 'rb') as mat_file:
        files = {
            'file': (os.path.basename(mat_file_path), mat_file, 'application/octet-stream')
        }
        
        # Send the request
        print('------')
        print(api_url)
        print('------')
        response = requests.post(
            api_url,
            files=files
        )
    
    # Process the response
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

def create_sample_mat_file(output_path, shape=(10, 10, 10)):
    """
    Create a sample .mat file with random 3D volume data
    
    Args:
        output_path (str): Path to save the .mat file
        shape (tuple): Shape of the 3D volume (default: (10, 10, 10))
    """
    print(f"Creating sample .mat file at {output_path}...")
    
    # Create a random 3D volume
    sample_volume = np.random.random(shape)
    
    # Save to .mat file
    scipy.io.savemat(output_path, {'phs_tissue': sample_volume})
    
    print(f"Sample .mat file created with shape {shape}")
    return sample_volume

def display_result(result):
    """
    Display the API result
    
    Args:
        result (dict): API response
    """
    if result is None:
        print("No result to display")
        return
    
    print("\nAPI Response:")
    print(f"Status: {result['status']}")
    
    if result['status'] == 'success':
        result_array = np.array(result['result'])
        print(f"Result shape: {result_array.shape}")
        print(f"Result min: {np.min(result_array)}")
        print(f"Result max: {np.max(result_array)}")
        print(f"Result mean: {np.mean(result_array)}")
    else:
        print(f"Error message: {result.get('message', 'No error message')}")

def main():
    parser = argparse.ArgumentParser(description='QSMnet+ API Client Example')
    parser.add_argument('--url', type=str, default='http://localhost:5000/predict',
                        help='API endpoint URL (default: http://localhost:5000/predict)')
    parser.add_argument('--method', type=str, choices=['json', 'mat'], default='json',
                        help='Request method (json or mat) (default: json)')
    parser.add_argument('--mat-file', type=str, default='sample_volume.mat',
                        help='Path to .mat file for mat method (default: sample_volume.mat)')
    parser.add_argument('--create-sample', action='store_true',
                        help='Create a sample .mat file')
    parser.add_argument('--shape', type=int, nargs=3, default=[10, 10, 10],
                        help='Shape of sample volume (default: 10 10 10)')
    
    args = parser.parse_args()
    
    # Create sample .mat file if requested
    if args.create_sample:
        sample_volume = create_sample_mat_file(args.mat_file, tuple(args.shape))
    
    # Send the request based on the chosen method
    if args.method == 'json':
        # If we created a sample, use it, otherwise create a new random volume
        if args.create_sample:
            volume_data = sample_volume
        else:
            volume_data = np.random.random(tuple(args.shape))
        
        result = send_json_request(args.url, volume_data)
        display_result(result)
    else:  # mat method
        result = send_mat_file_request(args.url, args.mat_file)
        display_result(result)

if __name__ == "__main__":
    main()
