from httpx_auth import OAuth2ResourceOwnerPasswordCredentials
from wns_api_clients import Client as ApiClient
from wns_api_clients.api.cameras import read_camera
import os


def get_start_date(camera_id):

    # create objet ApiCLient
    api_url: str = "https://app.wavesnsee.com"
    api_client = ApiClient(
            base_url=api_url,
            httpx_args={
                "auth": OAuth2ResourceOwnerPasswordCredentials(
                    token_url=f"{api_url}/api/auth/access-token",
                    username=os.getenv('user_wns_api_client'),
                    password=os.getenv('passwd_wns_api_client'),
                ),
            },
        )

    resp_api = read_camera.sync(client=api_client,
                                camera_id=camera_id)

    start_date = resp_api.start_date
    start_date = start_date.replace(tzinfo=None)

    return start_date
