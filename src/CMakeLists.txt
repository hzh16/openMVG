
# Copyright (c) 2012, 2013 openMVG authors.

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

CMAKE_MINIMUM_REQUIRED(VERSION 3.9)

project(openMVG C CXX)

# Run automoc on .hh files in new versions of cmake
if (${CMAKE_VERSION} VERSION_GREATER "3.17" OR
    ${CMAKE_VERSION} VERSION_EQUAL "3.17")
  cmake_policy(SET CMP0100 NEW)
endif()

include(GNUInstallDirs)

# guard against in-source builds
if (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "In-source builds not allowed.")
endif()

set(CMAKE_FIND_FRAMEWORK LAST)

# ==============================================================================
# OpenMVG build options
# ==============================================================================
option(OpenMVG_BUILD_SHARED "Build OpenMVG shared libs" OFF)
option(OpenMVG_BUILD_TESTS "Build OpenMVG tests" OFF)
option(OpenMVG_BUILD_DOC "Build OpenMVG documentation" ON)
option(OpenMVG_BUILD_EXAMPLES "Build OpenMVG samples applications." ON)
option(OpenMVG_BUILD_OPENGL_EXAMPLES "Build OpenMVG openGL examples" OFF)
option(OpenMVG_BUILD_SOFTWARES "Build OpenMVG softwares" ON)
option(OpenMVG_BUILD_GUI_SOFTWARES "Build OpenMVG GUI softwares (QT5)" ON)
option(OpenMVG_BUILD_COVERAGE "Enable code coverage generation (gcc only)" OFF)
option(OpenMVG_USE_OPENMP "Enable OpenMP parallelization" ON)
# ==============================================================================
# Opencv is not used by openMVG but some samples show how to use openCV
#  and openMVG simultaneously
# ==============================================================================
option(OpenMVG_USE_OPENCV "Build or not opencv+openMVG samples programs" OFF)

# ==============================================================================
# Since OpenCV 3, SIFT is no longer in the default modules. See
# https://github.com/itseez/opencv_contrib for more informations.
# Enable this to be able to use OpenCV SIFT in main_ComputeFeatures_OpenCV.
# ==============================================================================
option(OpenMVG_USE_OCVSIFT "Add or not OpenCV SIFT in available features" OFF)

# ==============================================================================
# Since the LiGT algorithm has a patent protection.
# See header files in openMVG/multiview/LiGT for more informations.
#
# [Conditions of Use] The LiGT algorithm is distributed under the License of
# Attribution-Share Alike 4.0 International (https://creativecommons.org/licenses/by-sa/4.0/).
#
# Enable this to be able to use LiGT in main_SfM.
# ==============================================================================
option(OpenMVG_USE_LIGT "Add or not LiGT in available" ON)

# ==============================================================================
# OpenMVG version
# ==============================================================================
file(STRINGS "openMVG/version.hpp" _OPENMVG_VERSION_HPP_CONTENTS REGEX "#define OPENMVG_VERSION_")
foreach(v MAJOR MINOR REVISION)
  if("${_OPENMVG_VERSION_HPP_CONTENTS}" MATCHES "#define OPENMVG_VERSION_${v} ([0-9]+)")
    set(OPENMVG_VERSION_${v} "${CMAKE_MATCH_1}")
 else()
    message(FATAL_ERROR "Failed to retrieve the OpenMVG version from the source code. Missing OPENMVG_VERSION_${v}.")
  endif()
endforeach()
set(OPENMVG_VERSION ${OPENMVG_VERSION_MAJOR}.${OPENMVG_VERSION_MINOR}.${OPENMVG_VERSION_REVISION})

# Default build is in Release mode
if (NOT CMAKE_BUILD_TYPE AND NOT MSVC)
  set(CMAKE_BUILD_TYPE "Release")
endif (NOT CMAKE_BUILD_TYPE AND NOT MSVC)

# Set build path
set(EXECUTABLE_OUTPUT_PATH "${PROJECT_BINARY_DIR}/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}-${CMAKE_BUILD_TYPE}")
set(LIBRARY_OUTPUT_PATH "${PROJECT_BINARY_DIR}/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}-${CMAKE_BUILD_TYPE}")

# ==============================================================================
# Macro used to forward compiler preprocessors to OpenMVGConfig.cmake targets
# ==============================================================================
set(OpenMVG_DEFINITIONS "")
MACRO(register_definitions DEF)
  add_definitions(${DEF})
  string(REPLACE "-D" "" STRIPPED_DEF ${DEF})
  list(APPEND OpenMVG_DEFINITIONS ${STRIPPED_DEF})
ENDMACRO()

# Set shared or static mode
if (OpenMVG_BUILD_SHARED)
  set(BUILD_SHARED_LIBS ON)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  if(WIN32)
    # Avoid to use the dllexport markup
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
  endif()
else (OpenMVG_BUILD_SHARED)
  set(BUILD_SHARED_LIBS OFF)
endif()

# ==============================================================================
# Check that submodule have been initialized and updated
# ==============================================================================
if (NOT EXISTS ${PROJECT_SOURCE_DIR}/dependencies/cereal/include)
  message(FATAL_ERROR
    "\n submodule(s) are missing, please update your repository:\n"
    "  > git submodule update -i\n")
endif()

# ==============================================================================
# Additional cmake find modules
# ==============================================================================
set(CMAKE_MODULE_PATH
  ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmakeFindModules)
include(OptimizeForArchitecture)
OptimizeForArchitecture()

if (UNIX AND NOT OpenMVG_BUILD_COVERAGE)
  set(CMAKE_C_FLAGS_RELEASE "-O3")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3")
endif ()

# ==============================================================================
# Check C++11 support
# ==============================================================================
include(CXX11)
check_for_cxx11_compiler(CXX11_COMPILER)
# If a C++11 compiler is available, then set the appropriate flags
if (CXX11_COMPILER)
  set(CMAKE_CXX_STANDARD 11)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
else (CXX11_COMPILER)
  message(FATAL_ERROR "A CXX11 compiler is required.")
endif (CXX11_COMPILER)

# ==============================================================================
# OpenMP detection
# ==============================================================================
if (OpenMVG_USE_OPENMP)
  find_package(OpenMP)
  if (OPENMP_FOUND)
    register_definitions(-DOPENMVG_USE_OPENMP)
    include_directories(${OpenMP_C_INCLUDE_DIR})
    if(NOT TARGET OpenMP::OpenMP_CXX)
      find_package(Threads REQUIRED)
      add_library(OpenMP::OpenMP_CXX IMPORTED INTERFACE)
      set_property(TARGET OpenMP::OpenMP_CXX
                   PROPERTY INTERFACE_COMPILE_OPTIONS ${OpenMP_CXX_FLAGS})
      # Only works if the same flag is passed to the linker; use CMake 3.9+ otherwise (Intel, AppleClang)
      set_property(TARGET OpenMP::OpenMP_CXX
                   PROPERTY INTERFACE_LINK_LIBRARIES ${OpenMP_CXX_FLAGS} Threads::Threads)
    endif()
    if( ${OpenMP_CXX_INCLUDE_DIRS} )
      include_directories( ${OpenMP_CXX_INCLUDE_DIRS} )
    else()
      # Macports fix
      #   find_package(OpenMP) is able to find the openMP dylib but fails to find the include dir,
      #   we force it by hand.
      if( APPLE AND EXISTS "/opt/local/lib/libomp" AND EXISTS "/opt/local/include/libomp" )
        set_property( TARGET OpenMP::OpenMP_CXX PROPERTY INTERFACE_INCLUDE_DIRECTORIES "/opt/local/include/libomp" )
        include_directories( "/opt/local/include/libomp")
      endif()
    endif()
    list(APPEND OPENMVG_LIBRARY_DEPENDENCIES OpenMP::OpenMP_CXX)
  endif (OPENMP_FOUND)
else (OpenMVG_USE_OPENMP)
    option(OpenMVG_USE_OPENMP "Use OpenMP for parallelization" OFF)
    include(UpdateCacheVariable)
    UPDATE_CACHE_VARIABLE(OpenMVG_USE_OPENMP OFF)
    remove_definitions(-DOPENMVG_USE_OPENMP)
endif (OpenMVG_USE_OPENMP)

# ==============================================================================
# enable code coverage generation (only with GCC)
# ==============================================================================
if (OpenMVG_BUILD_COVERAGE AND CMAKE_COMPILER_IS_GNUCXX)
  message("OpenMVG_BUILD_COVERAGE enabled")
  set(CMAKE_BUILD_TYPE "Debug")
  add_definitions(--coverage -fprofile-arcs -ftest-coverage)
  set(CMAKE_EXE_LINKER_FLAGS
    "${CMAKE_EXE_LINKER_FLAGS} -fprofile-arcs -ftest-coverage")
endif()

# ==============================================================================
# IMAGE IO detection
# ==============================================================================
find_package(JPEG QUIET)
find_package(PNG QUIET)
find_package(TIFF QUIET)

# Folders
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# ==============================================================================
# SUBMODULE CONFIGURATION
# ==============================================================================
#- glfw
# ==============================================================================
if (OpenMVG_BUILD_OPENGL_EXAMPLES)
  set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "Do not build the GLFW example programs")
  set(GLFW_BUILD_TESTS    OFF CACHE BOOL "Do not build the GLFW tests programs")
  set(GLFW_BUILD_DOCS     OFF CACHE BOOL "Do not build the GLFW documentation")
  set(GLFW_INSTALL        OFF CACHE BOOL "Do not generate the GLFW installation target")
  add_subdirectory(dependencies/glfw)
  include_directories(SYSTEM dependencies/glfw/include)
  set_property(TARGET glfw PROPERTY FOLDER OpenMVG/3rdParty/glfw)
endif (OpenMVG_BUILD_OPENGL_EXAMPLES)

# Dependencies install rules
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/
  DESTINATION include/openMVG_dependencies
  COMPONENT headers
  FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h"
)

# ==============================================================================
# --END-- SUBMODULE CONFIGURATION
# ==============================================================================

# ==============================================================================
# Enable cmake UNIT TEST framework
# ==============================================================================
if (OpenMVG_BUILD_TESTS)
  enable_testing()
endif (OpenMVG_BUILD_TESTS)
# MACRO to ease UnitTesting
MACRO(UNIT_TEST NAMESPACE NAME EXTRA_LIBS)
  if (OpenMVG_BUILD_TESTS)
    add_executable(${NAMESPACE}_test_${NAME} ${NAME}_test.cpp)

    set_property(TARGET ${NAMESPACE}_test_${NAME} PROPERTY FOLDER OpenMVG/test)
    target_include_directories(${NAMESPACE}_test_${NAME}
                               PRIVATE
                               ${CMAKE_SOURCE_DIR}/third_party)
    target_link_libraries(${NAMESPACE}_test_${NAME}
                          ${EXTRA_LIBS} # Extra libs MUST be first.
                          CppUnitLite)
    add_test(NAME ${NAMESPACE}_test_${NAME} WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
             COMMAND $<TARGET_FILE:${NAMESPACE}_test_${NAME}>)
  target_compile_definitions(${NAMESPACE}_test_${NAME}
    PRIVATE -DTHIS_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  endif (OpenMVG_BUILD_TESTS)
ENDMACRO(UNIT_TEST)

# ==============================================================================
# cereal
# ==============================================================================
# - external by default
# - internal if cereal not found
# - as of cereal:1.3.1 target requires component
# ==============================================================================
find_package(cereal QUIET CONFIG)
if (NOT cereal_FOUND)
  add_library(cereal INTERFACE)
  target_include_directories(cereal
    INTERFACE
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/dependencies/cereal/include>
      $<INSTALL_INTERFACE:include/openMVG_dependencies/cereal/include>
  )
  install(TARGETS cereal EXPORT openMVG-targets)
  set(OpenMVG_USE_INTERNAL_CEREAL ON)
endif()
if (TARGET cereal::cereal)
  set(cereal_TARGET "cereal::cereal")
else()
  set(cereal_TARGET "cereal")
endif()

# ==============================================================================
# Eigen
# ==============================================================================
# - external by default
# - external if EIGEN_INCLUDE_DIR_HINTS is defined
# - internal if Eigen not found
# ==============================================================================
#find_package(Eigen3 QUIET)
if (NOT Eigen3_FOUND)
  set(EIGEN_INCLUDE_DIR_HINTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen)
  set(OpenMVG_USE_INTERNAL_EIGEN ON)
  find_package(Eigen QUIET)
else()
  set(EIGEN_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})
endif()
add_definitions(-DEIGEN_MPL2_ONLY)

# ==============================================================================
# Ceres
# ==============================================================================
# - external by default if CERES_DIR_HINTS or find_package found a valid Ceres
# - internal if ceres not found (ceres-solver+cxsparse+miniglog)
# ==============================================================================
#find_package(Ceres QUIET HINTS ${CERES_DIR_HINTS})
if (NOT Ceres_FOUND)
  set(OpenMVG_USE_INTERNAL_CERES ON)
  set(CERES_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ceres-solver/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ceres-solver/internal/ceres/miniglog
    ${PROJECT_BINARY_DIR}/third_party/ceres-solver/config)
    FILE(READ "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ceres-solver/VERSION" CERES_CONFIG)
    STRING(REGEX REPLACE "version ([0-9.]+).*" "\\1" CERES_VERSION ${CERES_CONFIG})
  set(CERES_LIBRARIES openMVG_ceres)
endif()

# ==============================================================================
# Flann
# ==============================================================================
# - internal by default (flann),
# - external if FLANN_INCLUDE_DIR_HINTS and a valid Flann setup is found
# ==============================================================================
if (NOT DEFINED FLANN_INCLUDE_DIR_HINTS)
  set(FLANN_INCLUDE_DIR_HINTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/flann/src/cpp)
  set(OpenMVG_USE_INTERNAL_FLANN ON)
  find_package(Flann QUIET)
else()
  find_package(Flann QUIET CONFIG)
  if (TARGET flann::flann)
    set(FLANN_LIBRARIES "flann::flann;flann::flann_cpp")
  else()
    find_package(Flann QUIET)
  endif()
endif()
if (NOT FLANN_FOUND OR OpenMVG_USE_INTERNAL_FLANN)
  set(FLANN_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/flann/src/cpp)
endif()

# ==============================================================================
# CoinUtils
# ==============================================================================
# - internal by default (CoinUtils),
# - external if COINUTILS_INCLUDE_DIR_HINTS and a valid CoinUtils setup is found
# ==============================================================================
if (NOT DEFINED COINUTILS_INCLUDE_DIR_HINTS)
  set(COINUTILS_INCLUDE_DIR_HINTS ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/osi_clp/CoinUtils/src/)
  set(OpenMVG_USE_INTERNAL_COINUTILS ON)
  find_package(CoinUtils QUIET)
  set(COINUTILS_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/osi_clp/CoinUtils/src/)
  set(COINUTILS_LIBRARY lib_CoinUtils)
else()
  find_package(CoinUtils QUIET)
endif()

## ==============================================================================
## Clp
## ==============================================================================
## - internal by default (Clp),
## - external if CLP_INCLUDE_DIR_HINTS and a valid Clp setup is found
## ==============================================================================
if (NOT DEFINED CLP_INCLUDE_DIR_HINTS)
  set(CLP_INCLUDE_DIR_HINTS ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/osi_clp/Clp/src/)
  set(OpenMVG_USE_INTERNAL_CLP ON)
  find_package(Clp QUIET)
  set(CLP_INCLUDE_DIRS
     ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/osi_clp/Clp/src/
     ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/osi_clp/Clp/src/OsiClp/)
  set(CLP_LIBRARIES lib_clp lib_OsiClpSolver)
else()
  find_package(Clp QUIET)
endif()

# ==============================================================================
# Osi
# ==============================================================================
# - internal by default (Osi),
# - external if OSI_INCLUDE_DIR_HINTS and a valid Osi setup is found
# ==============================================================================
if (NOT DEFINED OSI_INCLUDE_DIR_HINTS)
  set(OSI_INCLUDE_DIR_HINTS ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/osi_clp/Osi/src/)
  set(OpenMVG_USE_INTERNAL_OSI ON)
  find_package(Osi QUIET)
  set(OSI_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/osi_clp/Osi/src/Osi/)
  set(OSI_LIBRARY lib_Osi)
else()
  find_package(Osi QUIET)
endif()

# ==============================================================================
# Internal CLP/OSI/COINUTILS libraries:
# ==============================================================================
if (OpenMVG_USE_INTERNAL_OSI AND OpenMVG_USE_INTERNAL_CLP AND OpenMVG_USE_INTERNAL_COINUTILS)
  add_subdirectory(dependencies/osi_clp/)
  set_property(TARGET lib_clp PROPERTY FOLDER OpenMVG/3rdParty/lib_clp)
  set_property(TARGET lib_CoinUtils PROPERTY FOLDER OpenMVG/3rdParty/lib_CoinUtils)
  set_property(TARGET lib_Osi PROPERTY FOLDER OpenMVG/3rdParty/lib_Osi)
  set_property(TARGET lib_OsiClpSolver PROPERTY FOLDER OpenMVG/3rdParty/lib_OsiClpSolver)
endif()

# ==============================================================================
# Lemon
# ==============================================================================
# - internal by default (Lemon),
# - external if LEMON_INCLUDE_DIR_HINTS and a valid Lemon setup is found
# ==============================================================================
if (NOT DEFINED LEMON_INCLUDE_DIR_HINTS)
  set(LEMON_INCLUDE_DIR_HINTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/lemon)
  set(OpenMVG_USE_INTERNAL_LEMON ON)
endif()
find_package(Lemon QUIET)
if (NOT LEMON_FOUND OR OpenMVG_USE_INTERNAL_LEMON)
  set(LEMON_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/lemon
    ${PROJECT_BINARY_DIR}/third_party/lemon)
endif()

# ==============================================================================
# OpenCV
# ==============================================================================
# - only external and enabled only if OpenMVG_USE_OPENCV is set to ON
# ==============================================================================
if (OpenMVG_USE_OPENCV)
  find_package( OpenCV QUIET )
  if (NOT OpenCV_FOUND OR OpenCV_VERSION VERSION_LESS "3.0.0")
    message(STATUS "OpenCV was not found (note that OpenCV version >= 3.0.0 is required). -> Disabling OpenCV support.")
    UPDATE_CACHE_VARIABLE(OpenMVG_USE_OPENCV OFF)
    UPDATE_CACHE_VARIABLE(OpenMVG_USE_OCVSIFT OFF)
  endif()
endif()


# ==============================================================================
# LiGT
# ==============================================================================
# - enabled only if OpenMVG_USE_LIGT is set to ON
# ==============================================================================
if (OpenMVG_USE_LIGT)
    add_definitions(-DUSE_PATENTED_LIGT)
endif()


# ==============================================================================
# Third-party libraries:
# ==============================================================================
add_subdirectory(third_party)
add_subdirectory(testing)

# ==============================================================================
# openMVG modules
# ==============================================================================
# The openMVG library itself
add_subdirectory(openMVG)

# openMVG tutorial examples
if (OpenMVG_BUILD_EXAMPLES)
  add_subdirectory(openMVG_Samples)
endif (OpenMVG_BUILD_EXAMPLES)

# Complete software(s) build on openMVG libraries
if (OpenMVG_BUILD_SOFTWARES)
  add_subdirectory(software)
endif (OpenMVG_BUILD_SOFTWARES)

# Included for research purpose only
add_subdirectory(nonFree)

# ==============================================================================
# Documentation
# --------------------------
# Sphinx detection
# ==============================================================================
if (OpenMVG_BUILD_DOC)
  find_package(Sphinx)
  if (EXISTS ${SPHINX_EXECUTABLE})
    set(SPHINX_HTML_DIR "${CMAKE_CURRENT_BINARY_DIR}/htmlDoc")

    configure_file(
      "${CMAKE_CURRENT_SOURCE_DIR}/../docs/sphinx/rst/conf.py"
      "${CMAKE_CURRENT_BINARY_DIR}/conf.py"
      @ONLY)

    add_custom_target(doc ALL
      ${SPHINX_EXECUTABLE}
      -b html
      "${CMAKE_CURRENT_SOURCE_DIR}/../docs/sphinx/rst"
      "${SPHINX_HTML_DIR}"
      COMMENT "Building HTML documentation with Sphinx")

    set_property(TARGET doc PROPERTY FOLDER OpenMVG)

  else (EXISTS ${SPHINX_EXECUTABLE})
    message("Sphinx need to be installed to generate the sphinx documentation")
  endif (EXISTS ${SPHINX_EXECUTABLE})


  # ------
  # Doxygen detection
  # ======
  find_package(Doxygen)
  if (DOXYGEN_FOUND)

    configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/../docs/doxygen/Doxyfile.in
                    ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY )

    add_custom_target( doc_doxygen ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
                       WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                       COMMENT "Generating API documentation with Doxygen" VERBATIM )

  else (DOXYGEN_FOUND)
    message("Doxygen need to be installed to generate the doxygen documentation")
  endif (DOXYGEN_FOUND)
endif()



message("\n")

message("** OpenMVG version: " ${OPENMVG_VERSION})
message("** Build Shared libs: " ${OpenMVG_BUILD_SHARED})
message("** Build OpenMVG tests: " ${OpenMVG_BUILD_TESTS})
message("** Build OpenMVG softwares: " ${OpenMVG_BUILD_SOFTWARES})
message("** Build OpenMVG GUI softwares: " ${OpenMVG_BUILD_GUI_SOFTWARES})
message("** Build OpenMVG documentation: " ${OpenMVG_BUILD_DOC})
message("** Build OpenMVG samples applications: " ${OpenMVG_BUILD_EXAMPLES})
message("** Build OpenMVG openGL examples: " ${OpenMVG_BUILD_OPENGL_EXAMPLES})
message("** Enable code coverage generation: " ${OpenMVG_BUILD_COVERAGE})
message("** Enable OpenMP parallelization: " ${OpenMVG_USE_OPENMP})
message("** Build OpenCV+OpenMVG samples programs: " ${OpenMVG_USE_OPENCV})
message("** Use OpenCV SIFT features: " ${OpenMVG_USE_OCVSIFT})
message("** Use LiGT for global translation estimation: " ${OpenMVG_USE_LIGT})

message("\n")

if (DEFINED OpenMVG_USE_INTERNAL_CEREAL)
  message(STATUS "CEREAL: (internal)")
else()
  message(STATUS "CEREAL: (external)")
endif()

if (DEFINED OpenMVG_USE_INTERNAL_EIGEN)
  message(STATUS "EIGEN: " ${EIGEN_VERSION} " (internal)")
else()
  message(STATUS "EIGEN: " ${EIGEN_VERSION} " (external)")
endif()

if (DEFINED OpenMVG_USE_INTERNAL_CERES)
  message(STATUS "CERES: " ${CERES_VERSION} " (internal)")
else()
  message(STATUS "CERES: " ${CERES_VERSION} " (external)")
endif()

if (DEFINED OpenMVG_USE_INTERNAL_FLANN)
  message(STATUS "FLANN: " ${FLANN_VERSION} " (internal)")
else()
  message(STATUS "FLANN: " ${FLANN_VERSION} " (external)")
endif()

if (DEFINED OpenMVG_USE_INTERNAL_TIFF)
  message(STATUS "LIBTIFF: " ${TIFF_VERSION_STRING} " (internal)")
else()
  message(STATUS "LIBTIFF: " ${TIFF_VERSION_STRING} " (external)")
endif()

if (DEFINED OpenMVG_USE_INTERNAL_PNG)
  message(STATUS "LIBPNG: " ${PNG_VERSION_STRING} " (internal)")
else()
  message(STATUS "LIBPNG: " ${PNG_VERSION_STRING} " (external)")
endif()

if (DEFINED OpenMVG_USE_INTERNAL_JPEG)
  message(STATUS "LIBJPEG (internal)")
else()
  message(STATUS "LIBJPEG (external)")
endif()

if (DEFINED OpenMVG_USE_INTERNAL_CLP)
  message(STATUS "CLP: " ${CLP_VERSION} " (internal)")
else()
  message(STATUS "CLP: " ${CLP_VERSION} " (external)")
endif()

if (DEFINED OpenMVG_USE_INTERNAL_COINUTILS)
  message(STATUS "COINUTILS: " ${COINUTILS_VERSION} " (internal)")
else()
  message(STATUS "COINUTILS: " ${COINUTILS_VERSION} " (external)")
endif()

if (DEFINED OpenMVG_USE_INTERNAL_OSI)
  message(STATUS "OSI: " ${OSI_VERSION} " (internal)")
else()
  message(STATUS "OSI: " ${OSI_VERSION} " (external)")
endif()

if (DEFINED OpenMVG_USE_INTERNAL_LEMON)
  message(STATUS "LEMON: " ${LEMON_VERSION} " (internal)")
else()
  message(STATUS "LEMON: " ${LEMON_VERSION} " (external)")
endif()

message("\n")

# ==============================================================================
# INSTALL RULES
# ==============================================================================

install(EXPORT openMVG-targets
  FILE OpenMVGTargets.cmake
  NAMESPACE OpenMVG::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/openMVG/cmake
)


# Build OpenMVG_CFLAGS variable for export
set(OpenMVG_CFLAGS "")
foreach(d ${OpenMVG_DEFINITIONS})
  list(APPEND OpenMVG_CFLAGS "-D${d}")
endforeach()

# Create a OpenMVGConfig.cmake file. <name>Config.cmake files are searched by
# find_package() automatically. We configure that file so that we can put any
# information we want in it, e.g. version numbers, include directories, etc.
configure_file("${CMAKE_SOURCE_DIR}/cmakeFindModules/OpenMVGConfig.cmake.in"
               "${CMAKE_CURRENT_BINARY_DIR}/OpenMVGConfig.cmake" @ONLY)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/OpenMVGConfig.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/openMVG/cmake)

# ==============================================================================
# UNINSTALL RULES
# ==============================================================================

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
    IMMEDIATE @ONLY)

add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
