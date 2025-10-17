#pragma once
#include <pybind11/pybind11.h>

namespace py = pybind11;

py::dict read_flare(const std::string &flare_file);
void bind_flare(py::module_ &m);
