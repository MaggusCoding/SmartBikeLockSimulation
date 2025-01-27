cmake_minimum_required(VERSION 3.15)
project(SmartBikeLockSimulation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find FFTW3
find_package(FFTW3 REQUIRED)

# List all source files explicitly
set(SOURCES
    src/main.cpp
    src/NeuralNetwork/NeuralNetwork.cpp
    src/FeatureExtractor/FeatureExtractor.cpp
    src/DataLoader/DataLoader.cpp
    src/DataPreprocessor/DataPreprocessor.cpp
    src/Metrics/Metrics.cpp
    src/FederatedClient/FederatedClient.cpp
    src/FederatedServer/FederatedServer.cpp
    src/HPO/HyperParameterOptimizer.cpp
)

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Add include directories
target_include_directories(${PROJECT_NAME}
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${FFTW3_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(${PROJECT_NAME} 
    PRIVATE 
        m
        fftw3
        fftw3f
)

# Print debug info
message(STATUS "Source files: ${SOURCES}")
message(STATUS "Include directories: ${CMAKE_CURRENT_SOURCE_DIR}/include")