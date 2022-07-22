#include "pypnp.h"
#include "pystereo.h"

PYBIND11_MODULE(pymvgkit_estimation, m)
{
  mvgkit::python::add_pnp_module(m);
  mvgkit::python::add_stereo_module(m);
}
