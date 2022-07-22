from enum import IntEnum
from typing import Dict, Optional

from mvgkit.common.camera import Camera
from mvgkit.mapping.mapper import sfm
from mvgkit.streamer import StreamerBase


class AvailableMapperType(IntEnum):
    IncrementalSFM = 0


class MapperManager:
    @staticmethod
    def create_mapper(
        mapper_type: AvailableMapperType,
        streamer: StreamerBase,
        camera: Optional[Camera] = None,
        params: Optional[Dict] = None,
    ):
        """
        TODO: add custom configuration from input arguments for each mapper class.
        """
        try:
            mapper_type = AvailableMapperType(mapper_type)
        except ValueError as e:
            print(e)

        mapper_cls = None
        if mapper_type == AvailableMapperType.IncrementalSFM:
            mapper_cls = sfm.IncrementalSFM

        if mapper_cls is not None:
            return mapper_cls(streamer=streamer, camera=camera, params=params)
