# YAM Arm - Comandos Principales

## Setup
Instalar paquetes de ros2 control:
```bash
sudo apt install ros-humble-ros2-control
sudo apt install ros-humble-ros2-controllers
```
## Simulación

### Lanzar simulación básica
```bash
ros2 launch yam_arm_gazebo yam_ctrl.gazebo.launch.py
```

## Control del Brazo

### Ver controladores disponibles
```bash
ros2 control list_controllers
ros2 action list
```

### Mover brazo con acción
```bash
ros2 action send_goal /arm_controller/follow_joint_trajectory \
  control_msgs/action/FollowJointTrajectory "
{
  trajectory: {
    joint_names: ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'],
    points: [{
      positions: [0.5, -0.5, 0.5, 0.0, 0.0, 0.0],
      time_from_start: {sec: 3, nanosec: 0}
    }]
  }
}"
```

## Control del Gripper

### Abrir gripper
```bash
ros2 action send_goal /gripper_controller/gripper_cmd \
  control_msgs/action/GripperCommand "
{
  command: {
    position: 0.04,
    max_effort: 50.0
  }
}"
```

### Cerrar gripper
```bash
ros2 action send_goal /gripper_controller/gripper_cmd \
  control_msgs/action/GripperCommand "
{
  command: {
    position: 0.0,
    max_effort: 50.0
  }
}"
```

## Scripts

### Ejecutar script de prueba
```bash
ros2 run yam_arm_gazebo control_arm_cli.py
```

### Ver estados de las articulaciones
```bash
ros2 topic echo /joint_states
```

### Ver información de acciones
```bash
ros2 action info /arm_controller/follow_joint_trajectory
ros2 action info /gripper_controller/gripper_cmd
```

## Moveit

### Correr la simulación y moveit en un mismo script
```bash
ros2 launch yam_arm_moveit_config yam_arm_moveit_gazebo.launch.py
```