import gdown
import os

google_drive_url = "https://drive.google.com/file/d/1-5wSkGwCJlTtt3F2vWpurq-CnFwdkx5R/view?usp=sharing"

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)

output_path = f"{current_directory}/checkpoint.pth"
# print(output_path)

gdown.download(google_drive_url, output_path, quiet=False, fuzzy=True)
