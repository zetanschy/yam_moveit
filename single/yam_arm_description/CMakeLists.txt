cmake_minimum_required(VERSION 3.10.2)

project(yam_arm_description)

find_package(ament_cmake REQUIRED)
find_package(urdf REQUIRED)

# Install the mesh files from SO101/assets
install(
	DIRECTORY
		SO101/assets/
	DESTINATION
		share/${PROJECT_NAME}/meshes
	FILES_MATCHING PATTERN "*.stl"
)

# Install URDF/Xacro files
install(
	DIRECTORY
		urdf/
	DESTINATION
		share/${PROJECT_NAME}/urdf
	FILES_MATCHING PATTERN "*.urdf" PATTERN "*.xacro"
)

# Install other directories
install(
	DIRECTORY
		gazebo_meshes
		meshes
		config
		launch
	DESTINATION
		share/${PROJECT_NAME}
	OPTIONAL
)

ament_package()

