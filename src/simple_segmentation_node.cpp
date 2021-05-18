#include <ros/ros.h>
#include <string>
#include "ipa_dirt_detection_dataset_tools/simple_segmentation.h"


int main(int argc, char** argv)
{
	ros::init(argc, argv, "simple_segmentation");
	ros::NodeHandle pnh("~");

	// read parameters
	std::string base_path;
	pnh.param("base_path", base_path, std::string(""));
	std::cout << "base_path: " << base_path << std::endl;
	std::string source_image_path;
	pnh.param("source_image_path", source_image_path, std::string("dirt"));
	std::cout << "images_to_be_segmented_path: " << source_image_path << std::endl;
	std::string rename_img_to;
	pnh.param("rename_img_to", rename_img_to, std::string(""));
	std::cout << "rename_img_to: " << rename_img_to << std::endl;
	std::string cropped_image_path;
	pnh.param("segmented_dirt_cropped_path", cropped_image_path, std::string("dirt_segmented"));
	std::cout << "Path to save the segmented dirt after cropping: " << cropped_image_path << std::endl;
	std::string cropped_mask_path;
	pnh.param("segmented_dirt_cropped_mask_path", cropped_mask_path, std::string("dirt_segmented"));
	std::cout << "segmented_dirt_cropped_mask_path: " << cropped_mask_path << std::endl;
	double foreground_rectangle_canny1;
	pnh.param("foreground_rectangle_canny1", foreground_rectangle_canny1, 400.);
	std::cout << "foreground_rectangle_canny1: " << foreground_rectangle_canny1 << std::endl;
	double foreground_rectangle_canny2;
	pnh.param("foreground_rectangle_canny2", foreground_rectangle_canny2, 1250.);
	std::cout << "foreground_rectangle_canny2: " << foreground_rectangle_canny2 << std::endl;
	double foreground_rectangle_min_area;
	pnh.param("foreground_rectangle_min_area", foreground_rectangle_min_area, 0.15);
	std::cout << "foreground_rectangle_min_area: " << foreground_rectangle_min_area << std::endl;
	double foreground_rectangle_target_area;
	pnh.param("foreground_rectangle_target_area", foreground_rectangle_target_area, 0.25);
	std::cout << "foreground_rectangle_target_area: " << foreground_rectangle_target_area << std::endl;
	double foreground_rectangle_shape_threshold;
	pnh.param("foreground_rectangle_shape_threshold", foreground_rectangle_shape_threshold, 0.8);
	std::cout << "foreground_rectangle_shape_threshold: " << foreground_rectangle_shape_threshold << std::endl;
	int foreground_rectangle_additional_cropping;
	pnh.param("foreground_rectangle_additional_cropping", foreground_rectangle_additional_cropping, 20);
	std::cout << "foreground_rectangle_additional_cropping: " << foreground_rectangle_additional_cropping << std::endl;
	double segmentation_threshold_L_lower;
	pnh.param("segmentation_threshold_L_lower", segmentation_threshold_L_lower, 30.);
	std::cout << "segmentation_threshold_L_lower: " << segmentation_threshold_L_lower << std::endl;
	double segmentation_threshold_L_upper;
	pnh.param("segmentation_threshold_L_upper", segmentation_threshold_L_upper, 100.);
	std::cout << "segmentation_threshold_L_upper: " << segmentation_threshold_L_upper << std::endl;
	double segmentation_threshold_ab;
	pnh.param("segmentation_threshold_ab", segmentation_threshold_ab, 20.);
	std::cout << "segmentation_threshold_ab: " << segmentation_threshold_ab << std::endl;
	int crop_residual;
	pnh.param("crop_residual", crop_residual, 2);
	std::cout << "Border residual for cropping bounding box is: " << crop_residual << std::endl;

	// run segmentation
	ipa_dirt_detection_dataset_tools::SimpleSegmentation segment_dirt(base_path + source_image_path, rename_img_to, base_path + cropped_image_path,
			base_path + cropped_mask_path, foreground_rectangle_canny1, foreground_rectangle_canny2, foreground_rectangle_min_area, foreground_rectangle_target_area,
			foreground_rectangle_shape_threshold, foreground_rectangle_additional_cropping, segmentation_threshold_L_lower, segmentation_threshold_L_upper,
			segmentation_threshold_ab, crop_residual);
	segment_dirt.run();

	return 0;
}
