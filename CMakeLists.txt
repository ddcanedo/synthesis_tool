cmake_minimum_required(VERSION 2.8.3)
project(ipa_dirt_detection_dataset_tools)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
	cv_bridge
	roscpp
	rospy
)


find_package(Boost REQUIRED COMPONENTS filesystem system)
find_package(OpenCV REQUIRED)


###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
	INCLUDE_DIRS
		include
	LIBRARIES
	CATKIN_DEPENDS
		roscpp
		rospy
	DEPENDS
)


###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
	include
	${catkin_INCLUDE_DIRS}
	${Boost_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS}
)


## RGB image recording tool with support for different folders for different classes
add_executable(record_dataset_rgb
	src/record_dataset_rgb.cpp
)
add_dependencies(record_dataset_rgb ${catkin_EXPORTED_TARGETS})
target_link_libraries(record_dataset_rgb
	${catkin_LIBRARIES}
	${OpenCV_LIBRARIES}
	${Boost_LIBRARIES}
)


## simple segmentation of raw dirt or object images
add_executable(simple_segmentation
	src/simple_segmentation_node.cpp
	src/simple_segmentation.cpp
)
add_dependencies(simple_segmentation ${catkin_EXPORTED_TARGETS})
target_link_libraries(simple_segmentation
	${catkin_LIBRARIES}
	${Boost_LIBRARIES}
	${OpenCV_LIBS}
)


## image blender for creating realistic floor images with dirt, objects, and shadows
add_executable(image_blender
	src/image_blender_node.cpp
	src/image_blender.cpp
)
add_dependencies(image_blender ${catkin_EXPORTED_TARGETS})
target_link_libraries(image_blender
	${catkin_LIBRARIES}
	${Boost_LIBRARIES}
	${OpenCV_LIBS}
)


## labeling tool for tagging bounding boxes with text in images
add_executable(bounding_box_labeler
	src/bounding_box_labeler.cpp
)
add_dependencies(bounding_box_labeler ${catkin_EXPORTED_TARGETS})
target_link_libraries(bounding_box_labeler
	${catkin_LIBRARIES}
	${Boost_LIBRARIES}
	${OpenCV_LIBS}
)



### dataset_create
#add_executable(dataset_create
#	src/obsolete/dataset_create.cpp
#	src/image_blender.cpp
#	src/simple_segmentation.cpp
#)
#add_dependencies(dataset_create ${catkin_EXPORTED_TARGETS})
#target_link_libraries(dataset_create
#	${catkin_LIBRARIES}
#	${Boost_LIBRARIES}
#	${OpenCV_LIBS}
#)


#############
## Install ##
#############
# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables and/or libraries for installation
install(TARGETS record_dataset_rgb simple_segmentation image_blender bounding_box_labeler
	ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
	LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
	RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Mark cpp header files for installation
install(DIRECTORY include/${PROJECT_NAME}/
	DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
	FILES_MATCHING PATTERN "*.h"
)

## Mark other files for installation (e.g. launch and bag files, etc.)
install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
