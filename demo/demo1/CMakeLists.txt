#
cmake_minimum_required (VERSION 3.8)

add_executable (RTHDdemo "demo.cpp")
set_target_properties(RTHDdemo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../output
)
add_dependencies(RTHDdemo shader_ptx)
target_link_libraries(RTHDdemo PRIVATE
    optixBaseLib
    geometryLib
    cudaLib
    hausdorffLib
)
target_include_directories(RTHDdemo PUBLIC
    ../../libs
    ../../libs/geometry
    ../../libs/CUDAUtil
    ../../libs/hausdorff
)

get_property(SHADER_PTX_OUTPUTS GLOBAL PROPERTY SHADER_PTX_OUTPUTS)
set(PTX_DEST_DIR "$<TARGET_FILE_DIR:RTHDdemo>")
add_custom_command(TARGET RTHDdemo POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${PTX_DEST_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SHADER_PTX_OUTPUTS} "${PTX_DEST_DIR}"
    COMMAND_EXPAND_LISTS
    COMMENT "Copying PTX files to ${PTX_DEST_DIR}"
)
