import os
import zipfile
import urllib.request
import multiprocessing


files = [
    "data_tracking_velodyne.zip",
    "data_tracking_oxts.zip",
    "data_tracking_calib.zip",
]

location = "https://s3.eu-central-1.amazonaws.com/avg-kitti/"

dst_dir = "data/tracking"
os.makedirs(dst_dir, exist_ok=True)


def download(file):
    url = location + file
    dst_file = os.path.join(dst_dir, file)

    if not os.path.exists(dst_file):
        print("Downloading", url)
        urllib.request.urlretrieve(url, dst_file)
    else:
        print("Skipping")
        return

    print("Unzipping", url)

    with zipfile.ZipFile(dst_file, 'r') as zip_ref:
        zip_ref.extractall(dst_dir)

    print("Done", url)


def main():
    pool = multiprocessing.Pool(len(files))
    pool.map(download, files)

    print("Done!")

if __name__ == "__main__":
    main()
