from dprapi import DPRAPI
from s2eapi import S2EAPI
from azureapi import AzureAPI
from decouple import config
import argparse

AZURE_KEY = config('AZURE_KEY')
AZURE_ENDPOINT = config('AZURE_ENDPOINT')


def process_images(
    azure_key = AZURE_KEY,
    azure_endpoint = AZURE_ENDPOINT,
    raw_dir = './work_dir/raw',
    aligned_dir = './work_dir/aligned',
    results_dir = './work_dir/results',
    network_pkl_gcloud_id = '1IxRyfTf62KBjyc486JA5tGLVnFh_d4eO',
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

def main():
    parser = argparse.ArgumentParser(description='Project real-world images into StyleGAN2 latent space')
    parser.add_argument('--skip_projection', type=bool, default=False)
    parser.add_argument('--skip_dpr', type=bool, default=False)
    parser.add_argument('--skip_azure', type=bool, default=False)
    args = parser.parse_args()
    process_images(
        skip_projection=args.skip_projection,
        skip_dpr=args.skip_dpr,
        skip_azure=args.skip_azure
    )

if __name__ == '__main__':
    main()