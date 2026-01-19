cmake_minimum_required(VERSION 3.10.2)

project(rtop_description)

find_package(ament_cmake REQUIRED)
find_package(urdf REQUIRED)

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
		meshes
		config
		launch
	DESTINATION
		share/${PROJECT_NAME}
	OPTIONAL
)

ament_package()

