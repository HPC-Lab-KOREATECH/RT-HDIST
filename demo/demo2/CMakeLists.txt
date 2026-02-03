#
cmake_minimum_required (VERSION 3.8)

add_executable (DissertationTest "dissertation_test.cpp"
    "SamplingMethods/cudaFunctions.cu"
    "cuBQLbased/cuBQLversion_impl.cu"
)
set_target_properties(DissertationTest PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../output/dissertation
)
add_dependencies(DissertationTest shader_ptx)
target_link_libraries(DissertationTest PRIVATE
    optixBaseLib
    geometryLib
    cudaLib
    hausdorffLib
    cuBQL_cuda_float3
)
target_include_directories(DissertationTest PUBLIC
    ../../libs
    ../../libs/geometry
    ../../libs/CUDAUtil
    ../../libs/hausdorff
    SamplingMethods
    cuBQLbased
    ${cuBQL_SOURCE_DIR}
)

get_property(SHADER_PTX_OUTPUTS GLOBAL PROPERTY SHADER_PTX_OUTPUTS)
set(PTX_DEST_DIR "$<TARGET_FILE_DIR:DissertationTest>")
add_custom_command(TARGET DissertationTest POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${PTX_DEST_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SHADER_PTX_OUTPUTS} "${PTX_DEST_DIR}"
    COMMAND_EXPAND_LISTS
    COMMENT "Copying PTX files to ${PTX_DEST_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:cuBQL_cuda_float3>"
        "$<TARGET_FILE_DIR:DissertationTest>"
)
