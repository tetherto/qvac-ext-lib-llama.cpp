# common_test library for load_into_memory.h and uint8-buff-stream.h

set(TARGET llama-common-test)

# Create an interface library (header-only)
add_library(${TARGET} INTERFACE)

# Add the include directories
target_include_directories(${TARGET} INTERFACE 
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Set a compile definition to indicate this target provides the specific header
target_compile_definitions(${TARGET} INTERFACE LLAMA_COMMON_TEST_HEADERS)

# Set compile features
target_compile_features(${TARGET} INTERFACE cxx_std_17)

# Link with common library
target_link_libraries(${TARGET} INTERFACE common)
