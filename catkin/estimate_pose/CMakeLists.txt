cmake_minimum_required(VERSION 2.8.3)
project(estimate_pose)
find_package(catkin REQUIRED COMPONENTS
  fovis
)

find_package(Eigen3 REQUIRED)

catkin_package(
  INCLUDE_DIRS ../../src/ ${GLIB2_INCLUDE_DIRS}
  LIBRARIES estimate_pose
  CATKIN_DEPENDS fovis
  DEPENDS GLIB2
#  CFG_EXTRAS bot-core-includes.cmake
)


file(GLOB c_files ../../src/*.cpp)
file(GLOB h_files ../../src/*.hpp)

include_directories(../../src/ ${catkin_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS})

add_library(estimate_pose SHARED ${c_files})
target_link_libraries(estimate_pose ${catkin_LIBRARIES}  m)
set_property(TARGET estimate_pose APPEND_STRING PROPERTY COMPILE_FLAGS -Wno-unused-result )

install(TARGETS estimate_pose
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION})

install(FILES ${h_files}
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})
