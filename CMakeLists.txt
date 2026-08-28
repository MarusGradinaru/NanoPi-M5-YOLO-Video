cmake_minimum_required(VERSION 3.10)

project(yolov8_video)

set(TARGET_SOC "rk3576")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Rockchip libraries/helpers copied locally into this project
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty 3rdparty.out)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/utils utils.out)

# OpenCV: video input, GUI display, frame handling
find_package(OpenCV REQUIRED)

# pthread
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_executable(${PROJECT_NAME}
    main.cc
    postprocess.cc
    secondary_classifier.cc
    rknpu2/yolov8.cc
    rknpu2/yolov8_zero_copy.cc
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    OUTPUT_NAME "yolov8-video"
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LIBRKNNRT_INCLUDES}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME}
    imageutils
    fileutils
    imagedrawing
    ${LIBRKNNRT}
    ${OpenCV_LIBS}
    Threads::Threads
    dl
)

set(CMAKE_INSTALL_RPATH "$ORIGIN/lib")

install(TARGETS ${PROJECT_NAME} DESTINATION .)

install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/model/coco_80_labels_list.txt
    DESTINATION model
)

file(GLOB RKNN_FILES "${CMAKE_CURRENT_SOURCE_DIR}/model/*.rknn")
install(FILES ${RKNN_FILES} DESTINATION model)

