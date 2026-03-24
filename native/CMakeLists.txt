cmake_minimum_required(VERSION 3.18)

project(mts_media_helper LANGUAGES CXX)

if(APPLE)
    enable_language(OBJCXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(APPLE)
    set(CMAKE_OBJCXX_STANDARD 17)
    set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
    set(CMAKE_OBJCXX_EXTENSIONS OFF)
endif()

if(APPLE)
    set(MTS_MEDIA_HELPER_SOURCES
        src/mts_media_helper_macos.cpp
    )
elseif(WIN32)
    set(MTS_MEDIA_HELPER_SOURCES
        src/mts_media_helper.cpp
    )
else()
    set(MTS_MEDIA_HELPER_SOURCES
        src/mts_media_helper_macos.cpp
    )
endif()

add_executable(mts_media_helper
    ${MTS_MEDIA_HELPER_SOURCES}
)

add_library(mts_posterior_fusion SHARED
    src/mts_posterior_fusion.cpp
)

if(APPLE)
    add_library(mts_apple_runtime_shim SHARED
        src/mts_apple_runtime_shim.mm
    )
endif()

if(MSVC)
    target_compile_definitions(mts_media_helper PRIVATE UNICODE _UNICODE)
    target_compile_definitions(mts_media_helper PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(mts_media_helper PRIVATE /W4 /EHsc)
else()
    target_compile_options(mts_media_helper PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(mts_posterior_fusion PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(APPLE)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    find_library(METAL_FRAMEWORK Metal REQUIRED)
    find_library(MPS_FRAMEWORK MetalPerformanceShaders REQUIRED)
    target_link_libraries(
        mts_apple_runtime_shim
        PRIVATE
        ${FOUNDATION_FRAMEWORK}
        ${METAL_FRAMEWORK}
        ${MPS_FRAMEWORK}
    )
    target_compile_options(mts_apple_runtime_shim PRIVATE -Wall -Wextra -Wpedantic)
    set_target_properties(
        mts_apple_runtime_shim
        PROPERTIES
        OUTPUT_NAME "mts_apple_runtime_shim"
    )
endif()

install(TARGETS mts_media_helper RUNTIME DESTINATION bin)
install(TARGETS mts_posterior_fusion LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
if(APPLE)
    install(TARGETS mts_apple_runtime_shim LIBRARY DESTINATION lib)
endif()
