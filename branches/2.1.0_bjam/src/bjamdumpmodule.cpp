#include "boost/python.hpp"

namespace bp=boost::python;

BOOST_PYTHON_MODULE(dumpmodule){
    bp::scope().attr("SHARK_DOLPHIN_STUDIO") = (int)1;
}
