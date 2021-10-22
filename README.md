# synthesis_tool

Adapted from: https://github.com/baker-project/baker/tree/indigo_dev/ipa_dirt_detection_dataset_tools

# Installation
1. Install ROS: http://wiki.ros.org/ROS/Installation
2. Run the following commands: 
- cmake .
- make

# Synthesis Tool 
1. Change the base path in launch/image_blender_params.yaml (Optional: you can also change the parameters)
2. Go to src/image_blender.cpp:
- If you do not want artificial lighting, comment lines 136 to 144
- If you do not want some kind of artificial dirt, comment 131, 132 or 133
3. Run the tool with the following commands: 
- source devel/setup.bash
- roslaunch ipa_dirt_detection_dataset_tools image_blender.launch

# Results
1. The folder "test_tool" is an example for this base path that you can use, and it already has the required samples to run
2. Considering you use this folder, the results are stored on test_tool/blended_floor_images/
3. I provide some scripts to verify the annotations and to convert from YOLO format to binary masks (test_tool/blended_floor_images/binary_mask.py)
