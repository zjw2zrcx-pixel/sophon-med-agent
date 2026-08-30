cmake_minimum_required(VERSION 3.10)
project(Qwen3VL)

if (NOT DEFINED TARGET_ARCH)
    set(TARGET_ARCH pcie)
endif()

include_directories(${PROJECT_SOURCE_DIR}/include)

include_directories(/opt/sophon/libsophon-current/include)
link_directories(/opt/sophon/libsophon-current/lib)

add_definitions(--std=c++17 -fPIC -Wall -Werror)
find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development)
find_path(PYBIND11_INCLUDE_DIR pybind11/pybind11.h REQUIRED)
execute_process(
    COMMAND ${Python_EXECUTABLE} -c
            "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '.so')"
    OUTPUT_VARIABLE PYTHON_EXTENSION_SUFFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE)

# Build directly from the system pybind11 headers.  The pybind11 CMake package
# shipped by some Sophon SoC images is tied to the OS Python (often 3.8), even
# when FindPython has correctly selected the requested Python 3.10 virtualenv.
add_library(chat MODULE chat.cpp)
set_target_properties(chat PROPERTIES
    PREFIX ""
    SUFFIX "${PYTHON_EXTENSION_SUFFIX}")
target_include_directories(chat PRIVATE
    ${PYBIND11_INCLUDE_DIR}
    ${Python_INCLUDE_DIRS})
target_link_libraries(chat PUBLIC bmrt bmlib)
install(TARGETS chat DESTINATION python)

