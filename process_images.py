from dprapi import DPRAPI
from s2eapi import S2EAPI
from azureapi import AzureAPI
from decouple import config

AZURE_KEY = config('AZURE_KEY')
AZURE_ENDPOINT = config('AZURE_ENDPOINT')


def process_images(
    azure_key = AZURE_KEY,
    azure_endpoint = AZURE_ENDPOINT,
    raw_dir = './work_dir/raw',
    aligned_dir = './work_dir/aligned',
    results_dir = './work_dir/results',
    network_pkl_gcloud_id = '1xAL82ELuRciaYR5PDQYaH2MHy6-UbkY-',
    vgg_pkl_gcloud_id = '1ofUti9VTZf2zqqDocuXqGrrZHI2JKh8p',
    skip_projection = False,
    skip_dpr = False,
    skip_azure = False):
    s2e_api = S2EAPI(raw_dir, aligned_dir, results_dir, network_pkl_gcloud_id, vgg_pkl_gcloud_id)
    dpr_api = DPRAPI(aligned_dir, results_dir)
    azure_api = AzureAPI(aligned_dir, results_dir, azure_key, azure_endpoint)


    if not skip_projection:
        s2e_api.align()
        s2e_api.project()

    if not skip_dpr:
        dpr_api.predict_light()

    if not skip_azure:
        azure_api.infer_face_features()
    
