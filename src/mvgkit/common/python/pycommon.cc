#include "pycamera.h"
#include "pygeometry.h"
#include "pytransformation.h"

PYBIND11_MODULE(pymvgkit_common, m)
{
  mvgkit::python::add_camera_module(m);
  mvgkit::python::add_geometry_module(m);
  mvgkit::python::add_transformation_module(m);
}
