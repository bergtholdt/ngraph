# ******************************************************************************
# Copyright 2017-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

# Enable ExternalProject CMake module
include(ExternalProject)

#------------------------------------------------------------------------------
# ONNX.proto definition version
#------------------------------------------------------------------------------

set(ONNX_VERSION 1.3.0)

#------------------------------------------------------------------------------
# Download and install libonnx ...
#------------------------------------------------------------------------------

set(ONNX_GIT_REPO_URL https://github.com/onnx/onnx.git)
set(ONNX_GIT_BRANCH rel-${ONNX_VERSION})

# # The 'BUILD_BYPRODUCTS' arguments was introduced in CMake 3.2.
# if (${CMAKE_VERSION} VERSION_LESS 3.2)
#     ExternalProject_Add(
#             ext_onnx
#             PREFIX onnx
#             GIT_REPOSITORY ${ONNX_GIT_REPO_URL}
#             GIT_TAG ${ONNX_GIT_BRANCH}
#             INSTALL_COMMAND ""
#             UPDATE_COMMAND ""
#             CMAKE_ARGS -DONNX_GEN_PB_TYPE_STUBS=OFF
#                        -DProtobuf_PROTOC_EXECUTABLE=${Protobuf_PROTOC_EXECUTABLE}
#                        -DProtobuf_INCLUDE_DIR=${Protobuf_INCLUDE_DIR}
#                        -DPROTOBUF_LIBRARY=${Protobuf_LIBRARY}
#                        -DPROTOBUF_INCLUDE_DIR=${Protobuf_INCLUDE_DIR}
#                        -DPROTOBUF_SRC_ROOT_FOLDER=${Protobuf_SRC_ROOT_FOLDER}
#             TMP_DIR "${EXTERNAL_PROJECTS_ROOT}/onnx/tmp"
#             STAMP_DIR "${EXTERNAL_PROJECTS_ROOT}/onnx/stamp"
#             DOWNLOAD_DIR "${EXTERNAL_PROJECTS_ROOT}/onnx/download"
#             SOURCE_DIR "${EXTERNAL_PROJECTS_ROOT}/onnx/src"
#             BINARY_DIR "${EXTERNAL_PROJECTS_ROOT}/onnx/bin"
#             INSTALL_DIR "${EXTERNAL_PROJECTS_ROOT}/onnx"
#             EXCLUDE_FROM_ALL TRUE
#     )
# else()
#     if (${CMAKE_VERSION} VERSION_LESS 3.6)
#         ExternalProject_Add(
#                 ext_onnx
#                 PREFIX ext_onnx
#                 GIT_REPOSITORY ${ONNX_GIT_REPO_URL}
#                 GIT_TAG ${ONNX_GIT_BRANCH}
#                 INSTALL_COMMAND ""
#                 UPDATE_COMMAND ""
#                 CMAKE_ARGS -DONNX_GEN_PB_TYPE_STUBS=OFF
#                            -DProtobuf_PROTOC_EXECUTABLE=${Protobuf_PROTOC_EXECUTABLE}
#                            -DProtobuf_INCLUDE_DIR=${Protobuf_INCLUDE_DIR}
#                            -DPROTOBUF_LIBRARY=${Protobuf_LIBRARY}
#                            -DPROTOBUF_INCLUDE_DIR=${Protobuf_INCLUDE_DIR}
#                            -DPROTOBUF_SRC_ROOT_FOLDER=${Protobuf_SRC_ROOT_FOLDER}
#                 TMP_DIR "${EXTERNAL_PROJECTS_ROOT}/onnx/tmp"
#                 STAMP_DIR "${EXTERNAL_PROJECTS_ROOT}/onnx/stamp"
#                 DOWNLOAD_DIR "${EXTERNAL_PROJECTS_ROOT}/onnx/download"
#                 SOURCE_DIR "${EXTERNAL_PROJECTS_ROOT}/onnx/src"
#                 BINARY_DIR "${EXTERNAL_PROJECTS_ROOT}/onnx/bin"
#                 INSTALL_DIR "${EXTERNAL_PROJECTS_ROOT}/onnx"
#                 BUILD_BYPRODUCTS ${EXTERNAL_PROJECTS_ROOT}/onnx/bin/libonnx_proto.a
#                                  ${EXTERNAL_PROJECTS_ROOT}/onnx/bin/libonnx.a
#                 EXCLUDE_FROM_ALL TRUE
#         )
#     else()
#         # To speed things up prefer 'shallow copy' for CMake 3.6 and later
#         ExternalProject_Add(
#                 ext_onnx
#                 PREFIX ext_onnx
#                 GIT_REPOSITORY ${ONNX_GIT_REPO_URL}
#                 GIT_TAG ${ONNX_GIT_BRANCH}
#                 GIT_SHALLOW TRUE
#                 INSTALL_COMMAND ""
#                 UPDATE_COMMAND ""
#                 CMAKE_ARGS -DONNX_GEN_PB_TYPE_STUBS=OFF
#                            -DProtobuf_PROTOC_EXECUTABLE=${Protobuf_PROTOC_EXECUTABLE}
#                            -DProtobuf_LIBRARY=${Protobuf_LIBRARY}
#                            -DProtobuf_INCLUDE_DIR=${Protobuf_INCLUDE_DIR}
#                 TMP_DIR "${EXTERNAL_PROJECTS_ROOT}/onnx/tmp"
#                 STAMP_DIR "${EXTERNAL_PROJECTS_ROOT}/onnx/stamp"
#                 DOWNLOAD_DIR "${EXTERNAL_PROJECTS_ROOT}/onnx/download"
#                 SOURCE_DIR "${EXTERNAL_PROJECTS_ROOT}/onnx/src"
#                 BINARY_DIR "${EXTERNAL_PROJECTS_ROOT}/onnx/bin"
#                 INSTALL_DIR "${EXTERNAL_PROJECTS_ROOT}/onnx"
#                 BUILD_BYPRODUCTS ${EXTERNAL_PROJECTS_ROOT}/onnx/bin/libonnx_proto.a
#                                  ${EXTERNAL_PROJECTS_ROOT}/onnx/bin/libonnx.a
#                 EXCLUDE_FROM_ALL TRUE
#         )
#     endif()
# endif()

# # -----------------------------------------------------------------------------

# ExternalProject_Get_Property(ext_onnx SOURCE_DIR BINARY_DIR)
find_package(ONNX)

SET(ONNX_INCLUDE_DIR ${CMAKE_INSTALL_PREFIX}/include/onnx)

function(add_cloned_imported_target dst src)
    add_library(${dst} INTERFACE IMPORTED)
    foreach(name INTERFACE_LINK_LIBRARIES INTERFACE_INCLUDE_DIRECTORIES INTERFACE_COMPILE_DEFINITIONS INTERFACE_COMPILE_OPTIONS)
        get_property(value TARGET ${src} PROPERTY ${name} )
        set_property(TARGET ${dst} PROPERTY ${name} ${value})
    endforeach()
endfunction()

function(message_clone src)
    foreach(name INTERFACE_LINK_LIBRARIES INTERFACE_INCLUDE_DIRECTORIES INTERFACE_COMPILE_DEFINITIONS INTERFACE_COMPILE_OPTIONS)
        get_property(value TARGET ${src} PROPERTY ${name} )
        message(STATUS "${src} PROPERTY ${name} ${value}")
    endforeach()
endfunction()

# this does not work unfortunately
# usin recipy from https://github.com/conan-io/conan/issues/2125
# add_library(onnx::libonnx ALIAS onnx)
# add_library(onnx::libonnx_proto ALIAS onnx_proto)

if (NOT TARGET onnx::libonnx)
    add_cloned_imported_target(onnx::libonnx onnx)
    message_clone(onnx::libonnx)
endif()

if (NOT TARGET onnnx::libonnx_proto)
    add_cloned_imported_target(onnx::libonnx_proto onnx_proto)
    # expected to import without "onnx/"
    set_target_properties(onnx::libonnx_proto PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_INSTALL_PREFIX}/include/onnx)
    message_clone(onnx::libonnx_proto)
endif()

