cmake_minimum_required(VERSION 2.8)

set( CMAKE_PREFIX_PATH "/opt/opencv3/" ) # where kendall installed opencv 3.0.0


project(mujoco_example)

find_package(OpenCV REQUIRED)


set(MUJOCO_DIR "/opt/mujoco/") #set the mujoco path here

add_definitions(-std=c++11)

include_directories( ${MUJOCO_DIR} ${OpenCV_INCLUDE_DIRS} ${OPENNI_INCLUDE_DIRS} )

link_directories( ${MUJOCO_DIR} )

#message(STATUS "Including: " ${INCLUDE_DIRECTORIES} )
get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach(dir ${dirs})
   message(STATUS "dir='${dir}'")
endforeach()



add_executable(simulate src/simulate.cpp )
target_link_libraries(simulate mujoco glfw GL ${OpenCV_LIBRARIES})

add_executable(hello src/hello.cpp )
target_link_libraries(hello mujoco glfw GL ${OpenCV_LIBRARIES})

