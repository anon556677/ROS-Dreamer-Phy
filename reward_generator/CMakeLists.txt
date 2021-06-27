cmake_minimum_required(VERSION 2.8.3)
project(reward_generator)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  image_transport
  cv_bridge
  nav_msgs
  roscpp
  rospy
  sensor_msgs
  rl_server
  std_msgs
  tf
  message_generation
)

find_package(OpenCV REQUIRED)

add_message_files(
   FILES
   docker_reward.msg
   follower_reward.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  INCLUDE_DIRS include
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_executable(shore_follower src/shore_follower.cpp)
add_executable(shore_docker src/shore_docker.cpp)
add_executable(shore_docker_v2 src/shore_docker_v2.cpp)

add_dependencies(shore_follower reward_generator_generate_messages_cpp)
add_dependencies(shore_docker reward_generator_generate_messages_cpp)
add_dependencies(shore_docker_v2 reward_generator_generate_messages_cpp)

target_link_libraries(shore_follower
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
)

target_link_libraries(shore_docker
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
)

target_link_libraries(shore_docker_v2
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
)