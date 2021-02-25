import DPRAPI from dprapi
import S2EAPI from s2eapi
import AzureAPI from azure

KEY = "8ba735f8bc8743cdb3299577b050e555"

# This endpoint will be used in all examples in this quickstart.
ENDPOINT = "https://facerec1234.cognitiveservices.azure.com/"

def process_images(azure_key, azure_endpoint):
    dpr_api = DPRAPI('./work_dir/aligned', './work_dir/lights')
    s2e_api = S2EAPI('./work_dir/raw', './work_dir/aligned', './work_dir/projected', 'https://drive.google.com/uc?id=1IxRyfTf62KBjyc486JA5tGLVnFh_d4eO', 'https://drive.google.com/uc?id=1N2-m9qszOeVC9Tq77WxsLnuWwOedQiD2')
    azure_apid = AzureAPI('./work_dir/aligned', './work_dir/azure', azure_key, azure_endpoint)

if __name__ == '__main__':
    
