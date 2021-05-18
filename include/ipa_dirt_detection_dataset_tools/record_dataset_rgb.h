#ifndef DATASET_CREATE_H_
#define DATASET_CREATE_H_

#include <iostream>
#include <string>
#include <sstream>

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>

#include <opencv2/opencv.hpp>

//boost
#include <boost/filesystem.hpp>

namespace ipa_DatasetCreate
{
class DatasetCreate
{
protected:
	ros::NodeHandle node_handle_;

	ros::Subscriber camera_rgb_image_sub_;

	std::string base_path_;

	int current_floor_index_;
	int current_image_index_;

	void addZerosToFileName(std::stringstream& name, const int index);

	void selectNextFreeFloorIndex();
	void selectNextFreeImageIndex();

public:
	DatasetCreate(ros::NodeHandle node_handle);
	~DatasetCreate();
	void init();
	void imageCallback(const sensor_msgs::ImageConstPtr& msg);

private:
	struct bgr
	{
		uchar b; /**< Blue channel value. */
		uchar g; /**< Green channel value. */
		uchar r; /**< Red channel value. */
	};
	int frame_counter_;
};
}
;

namespace patch
{
template<typename T> std::string to_string(const T& n)
{
	std::ostringstream stm;
	stm << n;
	return stm.str();
}
}
;

#endif
