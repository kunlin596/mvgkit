"""This module implements the data model for Kitti dataset.

TODO: Generalize the common part of the data attributes and make it generic.

See http://www.cvlibs.net/datasets/kitti/setup.php for more information
about the sensor configurations.

Notes about calibration:
A (camera 0) and B (camera 1) are gray scale cameras, C (camera 2) and D (camera 3) are color cameras.
A is the center of origin. In the calibration file the transformation from LiDAR to camera 0,
T_AL and IMU/GPS to LiDAR T_LI are provided.

Here, frame (A), (B), (C), (D) are camera 0, 1, 2, 3, frame (L) is LiDAR frame and frame (I) is IMU/GPS frame.

The camera images are stored in the following directories:

  - 'image_00': left rectified grayscale image sequence
  - 'image_01': right rectified grayscale image sequence
  - 'image_02': left rectified color image sequence
  - 'image_03': right rectified color image sequence


The coordinate systems are defined the following way, where directions
are informally given from the drivers view, when looking forward onto
the road:

  - Camera:   x: right,   y: down,  z: forward
  - LiDAR:    x: forward, y: left,  z: up
  - IMU/GPS:  x: forward, y: right, z: down

Reference:
    [1] https://github.com/yanii/kitti-pcl/blob/master/KITTI_README.TXT
"""

import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import tqdm
from scipy.spatial.transform import Rotation
from mvg import basic, camera, image_processing


@dataclass
class IMUPacket:
    """
    The data structure includes IMU/GPS attributes, however for simplicity it's called IMU here.

    NOTE: The information below is copied from dataformat.txt shipped w/ Kitti.

        lat:            latitude of the oxts-unit (deg)
        lon:            longitude of the oxts-unit (deg)
        alt:            altitude of the oxts-unit (m)
        roll:           roll angle (rad), 0 = level, positive = left side up, range: -pi .. +pi
        pitch:          pitch angle (rad), 0 = level, positive = front down, range: -pi/2 .. +pi/2
        yaw:            heading (rad), 0 = east, positive = counter clockwise, range: -pi .. +pi
        vn:             velocity towards north (m/s)
        ve:             velocity towards east (m/s)
        vf:             forward velocity, i.e. parallel to earth-surface (m/s)
        vl:             leftward velocity, i.e. parallel to earth-surface (m/s)
        vu:             upward velocity, i.e. perpendicular to earth-surface (m/s)
        ax:             acceleration in x, i.e. in direction of vehicle front (m/s^2)
        ay:             acceleration in y, i.e. in direction of vehicle left (m/s^2)
        ay:             acceleration in z, i.e. in direction of vehicle top (m/s^2)
        af:             forward acceleration (m/s^2)
        al:             leftward acceleration (m/s^2)
        au:             upward acceleration (m/s^2)
        wx:             angular rate around x (rad/s)
        wy:             angular rate around y (rad/s)
        wz:             angular rate around z (rad/s)
        wf:             angular rate around forward axis (rad/s)
        wl:             angular rate around leftward axis (rad/s)
        wu:             angular rate around upward axis (rad/s)
        pos_accuracy:   velocity accuracy (north/east in m)
        vel_accuracy:   velocity accuracy (north/east in m/s)
        navstat:        navigation status (see navstat_to_string)
        numsats:        number of satellites tracked by primary GPS receiver
        posmode:        position mode of primary GPS receiver (see gps_mode_to_string)
        velmode:        velocity mode of primary GPS receiver (see gps_mode_to_string)
        orimode:        orientation mode of primary GPS receiver (see gps_mode_to_string)
    """

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
class KittiImage:
    data: np.ndarray
    timestamp: float

    def to_image(self):
        return image_processing.Image(self.data, self.timestamp)


@dataclass
class LiDARScan:
    """
    Here, data contains 4*num values, where the first 3 values correspond to
    x,y and z, and the last value is the reflectance information. All scans
    are stored row-aligned, meaning that the first 4 values correspond to the
    first measurement. Since each scan might potentially have a different
    number of points, this must be determined from the file size when reading
    the file, where 1e6 is a good enough upper bound on the number of values:

    // allocate 4 MB buffer (only ~130*4*4 KB are needed)
    int32_t num = 1000000;
    float *data = (float*)malloc(num*sizeof(float));

    // pointers
    float *px = data+0;
    float *py = data+1;
    float *pz = data+2;
    float *pr = data+3;

    // load point cloud
    FILE *stream;
    stream = fopen (currFilenameBinary.c_str(),"rb");
    num = fread(data,sizeof(float),num,stream)/4;
    for (int32_t i=0; i<num; i++) {
        point_cloud.points.push_back(tPoint(*px,*py,*pz,*pr));
        px+=4; py+=4; pz+=4; pr+=4;
    }
    fclose(stream);

    x,y and y are stored in metric (m) Velodyne coordinates.
    """

    points: np.ndarray
    reflectance: np.ndarray
    timestamp: float


class KittiDrive:
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
        filename = f"{KittiDrive._get_file_name_from_index(index)}.png"
        image = cv2.imread(str(path / filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return [image]

    @staticmethod
    def _read_lidar_scan(path: Path, index: int):
        filename = f"{KittiDrive._get_file_name_from_index(index)}.bin"
        data = np.fromfile(path / filename, dtype=np.float32).reshape(-1, 4)
        points = data[:, :3]
        reflectance = data[:, -1]
        return points, reflectance

    @staticmethod
    def _read_imu_measurements(path: Path, index: int):
        filename = f"{KittiDrive._get_file_name_from_index(index)}.txt"
        with open(path / filename, "r") as f:
            raw_data = list(map(float, f.readline().split(" ")))
            raw_data[-5:] = list(map(int, raw_data[-5:]))
            packet = IMUPacket(*raw_data)
            return [packet]

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
            datalist.append(data_cls(*data, self._timestamps[data_id][i]))
        return datalist

    def read_image(self, camera_id: int, indices: List[int], data_type: str = "sync"):
        data_id = "image_{:02d}".format(camera_id)
        images = self._read_data(
            self._path / data_type, data_id, self._read_image, KittiImage, indices
        )
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


@dataclass
class KittiCameraCalibration:

    unrectified_image_size: np.ndarray
    unrectified_camera_matrix: np.ndarray
    unrectified_distortion_coefficients: np.ndarray

    R: Rotation
    t: np.ndarray
    R_rectification: Rotation

    image_size: np.ndarray
    P: np.ndarray

    def get_camera(self):
        dcoeff = self.unrectified_distortion_coefficients
        return camera.Camera(
            K=camera.CameraMatrix.from_matrix(self.unrectified_camera_matrix),
            k=camera.RadialDistortionModel(dcoeff[0], dcoeff[1], dcoeff[4]),
            p=camera.TangentialDistortionModel(dcoeff[2], dcoeff[3]),
            T=basic.SE3(self.R, self.t),
        )


@dataclass
class KittiStereoCalibration:
    """
    calib_cam_to_cam.txt: Camera-to-camera calibration

        - S_xx: 1x2 size of image xx before rectification
        - K_xx: 3x3 calibration matrix of camera xx before rectification
        - D_xx: 1x5 distortion vector of camera xx before rectification
        - R_xx: 3x3 rotation matrix of camera xx (extrinsic)
        - T_xx: 3x1 translation vector of camera xx (extrinsic)
        - S_rect_xx: 1x2 size of image xx after rectification
        - R_rect_xx: 3x3 rectifying rotation to make image planes co-planar
        - P_rect_xx: 3x4 projection matrix after rectification

    TODO: Improve and use Camera class
    """

    timestamp: float
    corner_dist: float
    calibrations: Dict[int, KittiCameraCalibration]


@dataclass
class KittiLiDARCalibration:
    """
    calib_velo_to_cam.txt: Velodyne-to-camera registration

        - R: 3x3 rotation matrix
        - T: 3x1 translation vector
        - delta_f: deprecated
        - delta_c: deprecated
    """

    timestamp: float
    R: Rotation
    t: np.ndarray
    delta_f: np.ndarray
    delta_c: np.ndarray


@dataclass
class KittiIMUCalibration:
    """
    calib_imu_to_velo.txt: IMU/GPS-to-Velodyne registration

        - R: 3x3 rotation matrix
        - T: 3x1 translation vector
    """

    timestamp: float
    R: Rotation
    t: np.ndarray


class KittiCalibration:
    def __init__(self, path: Path):
        raw_imu_to_lidar = self._read_raw_imu_to_lidar(path / "calib_imu_to_velo.txt")
        raw_lidar_to_cam = self._read_raw_lidar_to_cam(path / "calib_velo_to_cam.txt")
        raw_cam_to_cam = self._read_raw_cam_to_cam(path / "calib_cam_to_cam.txt")

        self._imu_calibration = self._process_imu_to_lidar(raw_imu_to_lidar)
        self._lidar_calibration = self._process_lidar_to_cam(raw_lidar_to_cam)
        self._stereo_calibration = self._process_cam_to_cam(raw_cam_to_cam)

    @property
    def lidar_calibration(self):
        return self._lidar_calibration

    @property
    def stereo_calibration(self):
        return self._stereo_calibration

    @property
    def imu_calibration(self):
        return self._imu_calibration

    @staticmethod
    def _read_raw_calib_file(path: Path):
        """Read raw calibration file
        The first line is datetime, and the rest are calibration data.
        """
        data = dict()
        with open(path, "r") as f:
            lines = f.read().splitlines()
            key, value = lines[0].split(": ", 1)
            timestamp = datetime.strptime(value, "%d-%b-%Y %H:%M:%S").timestamp()
            data[key] = timestamp
            for line in lines[1:]:
                key, value = line.split(": ", 1)
                data[key] = np.fromstring(value, sep=" ", dtype=np.float32)
        return data

    @staticmethod
    def _process_lidar_to_cam(data):
        return KittiLiDARCalibration(
            timestamp=data["calib_time"],
            R=Rotation.from_matrix(data["R"].reshape(3, 3)),
            t=data["T"],
            delta_f=data["delta_f"],
            delta_c=data["delta_c"],
        )

    @staticmethod
    def _process_single_cam(data):
        return KittiCameraCalibration(
            unrectified_image_size=data["S"].astype(np.int32),
            unrectified_camera_matrix=data["K"].reshape(3, 3),
            unrectified_distortion_coefficients=data["D"].reshape(5),
            R=Rotation.from_matrix(data["R"].reshape(3, 3)),
            t=data["T"],
            image_size=data["S_rect"].astype(np.int32),
            R_rectification=Rotation.from_matrix(data["R_rect"].reshape(3, 3)),
            P=data["P_rect"].reshape(3, 4),
        )

    @staticmethod
    def _process_cam_to_cam(data):
        rawdata = dict()
        for k, v in data.items():
            components = k.split("_")
            if not components[-1].isnumeric():
                continue

            index = int(components[-1])
            var_name = "_".join(components[:-1])

            if index not in rawdata:
                rawdata[index] = dict()
            rawdata[index][var_name] = v

        calibrations = dict()
        for k, v in rawdata.items():
            calibrations[k] = KittiCalibration._process_single_cam(v)

        return KittiStereoCalibration(
            timestamp=data["calib_time"],
            corner_dist=float(data["corner_dist"]),
            calibrations=calibrations,
        )

    @staticmethod
    def _process_imu_to_lidar(data):
        return KittiIMUCalibration(
            timestamp=data["calib_time"],
            R=Rotation.from_matrix(data["R"].reshape(3, 3)),
            t=data["T"],
        )

    @staticmethod
    def _read_raw_cam_to_cam(path: Path):
        return KittiCalibration._read_raw_calib_file(path)

    @staticmethod
    def _read_raw_imu_to_lidar(path: Path):
        return KittiCalibration._read_raw_calib_file(path)

    @staticmethod
    def _read_raw_lidar_to_cam(path: Path):
        return KittiCalibration._read_raw_calib_file(path)
