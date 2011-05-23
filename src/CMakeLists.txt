find_package(PkgConfig REQUIRED)

#add_definitions(-Wall -msse2 -msse3 -O2)
#add_definitions(-Wall -g -O0)

add_executable(pose-estimator-test
               test.cpp         )

add_library(estimate-pose SHARED
    pose_estimator.cpp
    refine_motion_estimate.cpp)
set_target_properties(estimate-pose PROPERTIES SOVERSION 1)

pods_use_pkg_config_packages(pose-estimator-test
    opencv
    eigen3
    )

#find_package(Boost 1.36 COMPONENTS program_options filesystem REQUIRED)
find_package(Boost 1.36)
include_directories(${Boost_INCLUDE_DIRS})
target_link_libraries(pose-estimator-test estimate-pose ${Boost_LIBRARIES})

pods_install_pkg_config_file(estimate-pose
    LIBS -lestimate-pose
    REQUIRES eigen3
    VERSION 0.0.1)

pods_use_pkg_config_packages(estimate-pose eigen3)

pods_install_libraries(estimate-pose)

pods_install_headers(pose_estimator.hpp
    refine_motion_estimate.cpp
    DESTINATION estimate-pose)

pods_install_executables(pose-estimator-test)


