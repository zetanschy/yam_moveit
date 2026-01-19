# RTOP Description Package

This package contains the URDF/Xacro description for the RTOP bimanual robot system.

## Structure

- **rtop.urdf.xacro**: Main robot description file that combines two YAM arms
- **yam_arm.xacro**: Xacro macro defining a single YAM arm with camera
- **meshes/meshes_extra_rtop/**: Directory for additional RTOP-specific meshes
- **launch/**: Launch files for publishing robot description

## Robot Configuration

- **Base**: Central body/base link
- **Left Arm**: Positioned at -0.305m (left side)
- **Right Arm**: Positioned at +0.305m (right side)
- **Arm Spacing**: 0.610m between arm centers
- **Arm Height**: 1.0m above base

## Usage

### For Isaac Sim URDF Importer

To publish the robot description for Isaac Sim:

```bash
# Build the package first
cd ~/moveit_isaac_ws
colcon build --packages-select rtop_description
source install/setup.bash

# Launch the URDF publisher (recommended for Isaac Sim)
ros2 launch rtop_description urdf_publisher.launch.py use_sim_time:=true

# Or use the full robot state publisher
ros2 launch rtop_description robot_state_publisher.launch.py use_sim_time:=true
```

The robot description will be published on the `/robot_description` topic, which Isaac Sim's ROS 2 URDF importer can subscribe to.

### Manual Processing

To manually process the xacro file:

```bash
# Process the xacro file
ros2 run xacro xacro $(ros2 pkg prefix --share rtop_description)/urdf/rtop.urdf.xacro > rtop.urdf

# Or use directly with robot_state_publisher
ros2 run robot_state_publisher robot_state_publisher --ros-args \
  -p robot_description:="$(ros2 run xacro xacro $(ros2 pkg prefix --share rtop_description)/urdf/rtop.urdf.xacro)"
```

## Dependencies

- `yam_arm_description`: Provides base arm meshes and structure
- `xacro`: For processing xacro files
- `robot_state_publisher`: For publishing robot state

## Isaac Sim Integration

When using with Isaac Sim:

1. Start the URDF publisher:
   ```bash
   ros2 launch rtop_description urdf_publisher.launch.py use_sim_time:=true
   ```

2. In Isaac Sim, use the ROS 2 URDF Importer extension:
   - The importer will automatically subscribe to `/robot_description` topic
   - Make sure ROS 2 bridge is running if needed
   - The robot will be imported with both arms properly positioned
