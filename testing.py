import requests

api_url = 'https://api.api-ninjas.com/v1/imagetotext'

# Open the image in binary read mode
with open('herbal.jpg', 'rb') as image_file_descriptor:
    files = {'image': image_file_descriptor}

    # Send the POST request
    r = requests.post(api_url, files=files)

    # Check if the request was successful
    if r.status_code == 200:
        response = r.json()
        
        # Print only the text from the JSON response
        for item in response:
            print(item.get('text', 'No text found'))
    else:
        print(f"Error: {r.status_code}, {r.text}")
