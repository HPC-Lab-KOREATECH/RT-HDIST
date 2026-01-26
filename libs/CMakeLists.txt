add_library(optixBaseLib
    ../include/OptiX_Base.cpp
    GlobalParam.cpp
)
target_link_libraries(optixBaseLib PUBLIC
    optix_common
)
target_include_directories(optixBaseLib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ../../include
)

add_subdirectory(geometry)
add_subdirectory(CUDAUtil)
add_subdirectory(hausdorff)
