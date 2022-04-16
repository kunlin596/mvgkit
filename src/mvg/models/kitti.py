"""This module implements the data model for Kitti dataset.

TODO: Generalize the common part of the data attributes and make it generic.

See http://www.cvlibs.net/datasets/kitti/setup.php for more information
about the sensor configurations.
"""

import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

import cv2
import numpy as np
import tqdm


@dataclass
class IMUPacket:
    lat: float
    lon: float
    alt: float

    roll: float
    pitch: float
    yaw: float

    vn: float
    ve: float
    vf: float
    vl: float
    vu: float

    ax: float
    ay: float
    az: float

    af: float
    al: float
    au: float

    wx: float
    wy: float
    wz: float

    wf: float
    wl: float
    wu: float

    pos_accuracy: float
    vel_accuracy: float

    nav_stat: float
    num_sats: float
    pos_mode: float
    vel_mode: float
    ori_mode: float


@dataclass
class IMUMeasurements:
    data: IMUPacket
    timestamp: float


@dataclass
class Image:
    data: np.ndarray
    timestamp: float


@dataclass
class LiDARScan:
    data: np.ndarray
    timestamp: float


class KittiDataset:
    def __init__(self, path: Path, unzip=False):
        """
        `path` has the format: 2011_10_03_drive_0058.
        """

        self._path = path
        self._sync_path = self._path / "sync"
        self._extract_path = self._path / "extract"

        assert self._ensure_unzip(
            self._path, "sync", unzip
        ), f"unzip={unzip}, but sync data has not been unzipped, please unzip it manually."
        assert self._ensure_unzip(
            self._path, "extract", unzip
        ), f"unzip={unzip}, but extract data has not been unzipped, please unzip it manually."

        self._timestamps = dict()

        with open(self._path / "info.txt", "r") as f:
            self._num = int(f.readline())

    def _ensure_unzip(self, basepath, datatype, unzip):
        pathname = basepath.name
        output_path = basepath / datatype
        if not output_path.exists():
            if not unzip:
                return False
            print(
                f"Unzipping {pathname}_sync.zip to {output_path}, " "this might take a long time..."
            )
            with zipfile.ZipFile(basepath / f"{pathname}_sync.zip", "r") as zip_ref:
                for member in tqdm.tqdm(zip_ref.infolist(), "Extracing"):
                    member.filename = f"{datatype}/{'/'.join(member.filename.split('/')[2:])}"
                    zip_ref.extract(member, basepath)
        return True

    def _ensure_file_index(self, index: int):
        assert 0 <= index < self._num

    @staticmethod
    def _read_timestamps(path: Path):
        with open(path, "r") as f:
            timestamps = f.read().splitlines()

            if not all(timestamps):
                return None

            timestamps = list(
                map(
                    lambda timestamp: datetime.strptime(
                        timestamp[:-4], "%Y-%m-%d %H:%M:%S.%f"
                    ).timestamp(),
                    timestamps,
                )
            )
            return timestamps

    @staticmethod
    def _get_file_name_from_index(index: int):
        return "{:010d}".format(index)

    @staticmethod
    def _read_image(path: Path, index: int):
        filename = f"{KittiDataset._get_file_name_from_index(index)}.png"
        image = cv2.imread(str(path / filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    @staticmethod
    def _read_lidar_scan(path: Path, index: int):
        filename = f"{KittiDataset._get_file_name_from_index(index)}.txt"
        return np.fromfile(path / filename, dtype=np.float32)

    @staticmethod
    def _read_imu_measurements(path: Path, index: int):
        filename = f"{KittiDataset._get_file_name_from_index(index)}.txt"
        with open(path / filename, "r") as f:
            raw_data = list(map(float, f.readline().split(" ")))
            raw_data[-5:] = list(map(int, raw_data[-5:]))
            packet = IMUPacket(*raw_data)
            return packet

    def _read_data(self, data_rootpath, data_id, read_func, data_cls, indices):
        assert isinstance(indices, list), f"Argument indices={indices} should be a list!"
        if data_id not in self._timestamps:
            timestamps = self._read_timestamps(data_rootpath / data_id / "timestamps.txt")
            assert timestamps is not None, f"No valid timestamps for {data_id}!"
            self._timestamps[data_id] = np.asarray(timestamps)

        assert min(indices) >= 0, f"Min index out of range! indices={indices}, min=0."
        assert max(indices) < len(
            self._timestamps[data_id]
        ), f"Max index out of range! indices={indices}, max={len(self._timestamps[data_id])}."

        datalist = []
        for i in indices:
            self._ensure_file_index(i)
            data = read_func(data_rootpath / data_id / "data", i)
            datalist.append(data_cls(data, self._timestamps[data_id][i]))
        return datalist

    def read_image(self, camera_id: int, indices: List[int], data_type: str = "sync"):
        data_id = "image_{:02d}".format(camera_id)
        images = self._read_data(self._path / data_type, data_id, self._read_image, Image, indices)
        return images

    def read_lidar_scan(self, indices: List[int], data_type: str = "sync"):
        scan = self._read_data(
            self._path / data_type, "velodyne_points", self._read_lidar_scan, LiDARScan, indices
        )
        return scan

    def read_imu_measurements(self, indices: List[int], data_type: str = "sync"):
        imu_measurements = self._read_data(
            self._path / data_type, "oxts", self._read_imu_measurements, IMUMeasurements, indices
        )
        return imu_measurements
