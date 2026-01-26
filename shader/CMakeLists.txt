# PTX file builder

set(SHADER_SOURCES
    __shader__hd__qcluster__.cu
    __shader__hd__sampling__.cu
)

if(NOT DEFINED CUDAToolkit_NVCC_EXECUTABLE)
    message(FATAL_ERROR "CUDAToolkit_NVCC_EXECUTABLE not set. Ensure CUDAToolkit is found in the top-level CMake.")
endif()

# Use the first CUDA arch from the project settings.
if(DEFINED CMAKE_CUDA_ARCHITECTURES)
    list(GET CMAKE_CUDA_ARCHITECTURES 0 CUDA_ARCH)
else()
    set(CUDA_ARCH 75)
endif()

set(PTX_OUTPUTS "")
set(SHADER_INCLUDE_DIRS
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../libs"
    "${CMAKE_CURRENT_SOURCE_DIR}/../libs/3rdParty"
    "${CMAKE_CURRENT_SOURCE_DIR}/../libs/hausdorff"
    "${CMAKE_CURRENT_SOURCE_DIR}/../include"
    "${CUDAToolkit_INCLUDE_DIRS}"
    "${OptiX_INCLUDE_DIR}"
)
set(SHADER_INCLUDE_FLAGS "")
foreach(INC_DIR IN LISTS SHADER_INCLUDE_DIRS)
    list(APPEND SHADER_INCLUDE_FLAGS "-I${INC_DIR}")
endforeach()
foreach(SHADER_FILE IN LISTS SHADER_SOURCES)
    get_filename_component(SHADER_NAME "${SHADER_FILE}" NAME_WE)
    set(PTX_FILE "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/${SHADER_NAME}.ptx")
    get_filename_component(PTX_DIR "${PTX_FILE}" DIRECTORY)
    add_custom_command(
        OUTPUT "${PTX_FILE}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${PTX_DIR}"
        COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}"
                --ptx "${CMAKE_CURRENT_SOURCE_DIR}/${SHADER_FILE}"
                -o "${PTX_FILE}"
                --gpu-architecture=sm_${CUDA_ARCH}
                ${SHADER_INCLUDE_FLAGS}
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${SHADER_FILE}"
        COMMENT "Building PTX ${PTX_FILE}"
        VERBATIM
        COMMAND_EXPAND_LISTS
    )
    list(APPEND PTX_OUTPUTS "${PTX_FILE}")
endforeach()

add_custom_target(shader_ptx ALL DEPENDS ${PTX_OUTPUTS})

set_property(GLOBAL PROPERTY SHADER_PTX_OUTPUTS "${PTX_OUTPUTS}")
