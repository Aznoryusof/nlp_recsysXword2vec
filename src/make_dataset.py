import urllib.request
import os

from google_drive_downloader import GoogleDriveDownloader as gdd

cwd = os.getcwd()


def _download_raw_dataset():
    gdd.download_file_from_google_drive(file_id='19ximkMm1UEkuZ2NUHohCzQpVLOZdzDFz',
                                        dest_path=os.path.join(cwd, "data\\raw\\data.csv"),
                                        unzip=False)

    return print("Raw dataset downloaded in data\\raw" + "\n")


def main():
    """ Download the datasets already prepared for this project.
    """
    _download_raw_dataset()

if __name__ == '__main__':
    main()