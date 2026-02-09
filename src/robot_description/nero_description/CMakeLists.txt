cmake_minimum_required(VERSION 3.5)
project(nero_description)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
install(
  DIRECTORY launch meshes rviz urdf config
  DESTINATION share/${PROJECT_NAME}
)
# 确保 launch 脚本被正确安装（避免 DIRECTORY 安装后文件为空）
install(
  FILES launch/gazebo_control.launch.py launch/rviz_gazebo.launch.py launch/ppo_sim2sim.launch.py
  DESTINATION share/${PROJECT_NAME}/launch
)
install(
  DIRECTORY scripts
  DESTINATION share/${PROJECT_NAME}
)
# 使脚本可通过 ros2 run nero_description <name> 运行
install(
  PROGRAMS scripts/publish_joint_positions.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME publish_joint_positions
)
install(
  PROGRAMS scripts/gain_joint_positions.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME gain_joint_positions
)
install(
  PROGRAMS scripts/spawn_reach_targets.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME spawn_reach_targets
)
install(
  PROGRAMS scripts/gazebo_ppo_play.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME gazebo_ppo_play
)
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
