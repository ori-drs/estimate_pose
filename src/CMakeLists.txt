find_package(PkgConfig REQUIRED)

#add_definitions(-Wall -msse2 -msse3 -O2)
#add_definitions(-Wall -g -O0)

add_executable(pose-estimator-test
               test.cpp
               )

add_library(estimate-pose SHARED
            pose_estimator.cpp
            )
        #refine_motion_estimate.cpp)
set_target_properties(estimate-pose PROPERTIES SOVERSION 1)

pods_use_pkg_config_packages(pose-estimator-test
    eigen3
    libfovis
    )

target_link_libraries(pose-estimator-test estimate-pose)

pods_install_pkg_config_file(estimate-pose
    LIBS -lestimate-pose
    REQUIRES eigen3 libfovis
    VERSION 0.0.1)

pods_use_pkg_config_packages(estimate-pose eigen3 libfovis)

pods_install_libraries(estimate-pose)

pods_install_headers(pose_estimator.hpp
                     DESTINATION estimate-pose)

pods_install_executables(pose-estimator-test)
