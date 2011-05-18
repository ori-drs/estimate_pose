find_package(PkgConfig REQUIRED)

#add_definitions(-Wall -msse2 -msse3 -O2)
#add_definitions(-Wall -g -O0)

add_executable(pose-estimator-test
               test.cpp
               pose_estimator.cpp
               refine_motion_estimate.cpp
              )

pods_use_pkg_config_packages(pose-estimator-test
    opencv
    eigen3
    )

#find_package(Boost 1.36 COMPONENTS program_options filesystem REQUIRED)
find_package(Boost 1.36)
include_directories(${Boost_INCLUDE_DIRS})
target_link_libraries(pose-estimator-test ${Boost_LIBRARIES})

pods_install_executables(pose-estimator-test)
