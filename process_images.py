from dprapi import DPRAPI
from s2eapi import S2EAPI
from azureapi import AzureAPI


def process_images(
    azure_key, 
    azure_endpoint,
    raw_dir = './work_dir/raw',
    aligned_dir = './work_dir/aligned',
    projected_dir = './work_dir/projected',
    lights_dir = './work_dir/lights',
    azure_dir = './work_dir/azure',
    netowrk_pkl = '1xAL82ELuRciaYR5PDQYaH2MHy6-UbkY-',
    vgg_pkl = '1ofUti9VTZf2zqqDocuXqGrrZHI2JKh8p'):
    s2e_api = S2EAPI(raw_dir, aligned_dir, projected_dir, netowrk_pkl, vgg_pkl)
    dpr_api = DPRAPI(aligned_dir, lights_dir)
    azure_api = AzureAPI(aligned_dir, azure_dir, azure_key, azure_endpoint)

    s2e_api.align()
    s2e_api.project()
    dpr_api.predict_light()
    azure_api.infer_face_features()
    
