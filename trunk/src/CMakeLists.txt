project(pyopencvext)
find_package(Boost COMPONENTS python REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PythonLibs REQUIRED)
file(GLOB SRCS pyopencvext/*)

include_directories(
	${Boost_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS}
	${PROJECT_SOURCE_DIR}
	${PROJECT_SOURCE_DIR}/pyopencvext
	${PROJECT_SOURCE_DIR}/pyopencvext/core
	${PROJECT_SOURCE_DIR}/pyopencvext/sdopencv
	${PROJECT_SOURCE_DIR}/pyopencvext/numpy_include
	${PYTHON_INCLUDE_PATH}
)

add_library(pyopencvext SHARED ${SRCS})

target_link_libraries(pyopencvext ${Boost_PYTHON_LIBRARY} ${PYTHON_LIBRARY} ${OpenCV_LIBS})
