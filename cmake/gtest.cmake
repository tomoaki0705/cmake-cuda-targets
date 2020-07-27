
include(ExternalProject)

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/third-party/include)

ExternalProject_Add(gtest_ext
    URL "https://github.com/google/googletest/archive/release-1.8.1.zip"
    BINARY_DIR "${CMAKE_BINARY_DIR}/third-party/gtest-build"
    SOURCE_DIR "${CMAKE_BINARY_DIR}/third-party/gtest-src"
    INSTALL_COMMAND "${CMAKE_COMMAND}"
        --build . 
	--target install
    CMAKE_ARGS "${gtest_cmake_args}"
        "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
        "-DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/third-party/"
)
set(GTEST_INCLUDE_DIRS
    "${CMAKE_BINARY_DIR}/third-party/gtest-src/include"
)
add_library(gtest STATIC IMPORTED)
set_target_properties(gtest PROPERTIES
    IMPORTED_LOCATION "${CMAKE_BINARY_DIR}/third-party/lib/libgtest.a"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_BINARY_DIR}/third-party/include"
)
add_dependencies(gtest gtest_ext)


enable_testing()

find_package(Threads)

function(cxx_test name sources)
    add_executable(${name} ${sources})
    target_link_libraries(${name} ${ARGN} gtest ${CMAKE_THREAD_LIBS_INIT})
    set_property(TARGET ${name} APPEND PROPERTY INCLUDE_DIRECTORIES
                 ${GTEST_INCLUDE_DIRS}
                 )
    # Working directory: where the dlls are installed.
    add_test(NAME ${name} 
             COMMAND ${name} "--gtest_break_on_failure")
endfunction()

