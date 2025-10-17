#pragma once
#include <pybind11/pybind11.h>

namespace py = pybind11;

py::dict read_rfmix(const std::string &msp_file);
void bind_rfmix(py::module_ &m);
