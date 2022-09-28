from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
import time
from glob import iglob


class DriveAuthenticator:
    def __init__(self):
        self.iniate_drive()

    def iniate_drive(self):
        auth.authenticate_user()
        gauth = GoogleAuth()
        gauth.credentials = GoogleCredentials.get_application_default()
        self.drive = GoogleDrive(gauth)

    def download_file(self, fid):
        downloaded = self.drive.CreateFile({'id': fid})
        downloaded.FetchMetadata(fetch_all=True)
        downloaded.GetContentFile(downloaded.metadata['title'])


def keepNB_awake():
    while True:
        time.sleep(5)


def list_files(dirpath, fl_sort=True):
    fpaths = list(iglob(dirpath + '/*'))
    if fl_sort:
        fpaths.sort()
    return fpaths
