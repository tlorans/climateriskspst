import gdown

# Google Drive file ID
file_id = "17wm7QOqy90s7Jy-v1ZEO-iOkZn4mdXP7"
# The output filename (you can change this as needed)
output = "data/data_ml.xlsx"

# Construct the direct download URL
url = f"https://drive.google.com/uc?id={file_id}"

# Download the file
gdown.download(url, output, quiet=False)
