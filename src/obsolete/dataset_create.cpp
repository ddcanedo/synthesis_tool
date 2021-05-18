#include "ipa_dirt_detection_dataset_tools/obsolete/dataset_create.h"


ipa_dirt_detection_dataset_tools::DatasetCreate::DatasetCreate(ros::NodeHandle nh) :
		nh_(nh)
{
}

void ipa_dirt_detection_dataset_tools::DatasetCreate::init()
{
//  nh_.param("black_white_background", black_white_background_, true);
//  std::cout << "The dirt frames backgroound is : " << black_white_background_ << std::endl;
	nh_.param("segment_dirt_or_blend", segment_dirt_or_blend_, true);
	std::cout << "segmentation or blending ? " << segment_dirt_or_blend_ << std::endl;

//  nh_.param("blended_img_num", blended_img_num_, 0);
//  std::cout << "Number of blended images to be generated is: " << blended_img_num_ << std::endl;

	nh_.param("ground_image_path", ground_image_path_, std::string("/home/rmb-jx/dataset_new/ground_image"));
	std::cout << "ground_image_path: " << ground_image_path_ << std::endl;

	nh_.param("images_to_be_segmented_path", dirts_to_segmented_path_, std::string("/home/rmb-jx/dataset_new/Dirt"));
	std::cout << "images_to_be_segmented_path: " << dirts_to_segmented_path_ << std::endl;

	nh_.param("segmented_dirt_cropped_path", segmented_dirt_cropped_path_, std::string("/home/rmb-jx/dataset_new/Dirt"));
	std::cout << "Path to save the segmented dirt after cropping: " << segmented_dirt_cropped_path_ << std::endl;
	nh_.param("segmented_dirt_cropped_mask_path", segmented_dirt_cropped_mask_path_, std::string("/home/rmb-jx/dataset_new/Dirtmask"));
	std::cout << "segmented_dirt_cropped_mask_path: " << segmented_dirt_cropped_mask_path_ << std::endl;

	nh_.param("segmented_objects_path", segmented_pens_path_, std::string("/home/rmb-jx/dataset_new/Object"));
	std::cout << "segmented_objects_path: " << segmented_pens_path_ << std::endl;
	nh_.param("segmented_objects_mask_path", segmented_pens_mask_path_, std::string("/home/rmb-jx/dataset_new/Objectmask"));
	std::cout << "segmented_objects_mask_path: " << segmented_pens_mask_path_ << std::endl;

	nh_.param("blended_ground_image_path", blended_ground_image_path_, std::string("/home/rmb-jx/dataset_new/blended_ground_image"));
	std::cout << "blended_ground_image_path: " << blended_ground_image_path_ << std::endl;
	nh_.param("blended_ground_image_mask_path", blended_ground_image_mask_path_, std::string("/home/rmb-jx/dataset_new/blended_ground_mask"));
	std::cout << "blended_ground_image_mask_path: " << blended_ground_image_mask_path_ << std::endl;

	nh_.param("crop_residual", crop_residual_, 2);
	std::cout << "Border residual for cropping bounding box is: " << crop_residual_ << std::endl;

	nh_.param("max_num_dirts", max_num_dirts_, 8);
	std::cout << "Maximum number of dirts per frame: " << max_num_dirts_ << std::endl;
	nh_.param("min_num_dirts", min_num_dirts_, 5);
	std::cout << "Minimum number of dirts per frame: : " << min_num_dirts_ << std::endl;

	nh_.param("max_num_objects", max_num_pens_, 6);
	std::cout << "Maximum number of pens per frame: " << max_num_pens_ << std::endl;
	nh_.param("min_num_objects", min_num_pens_, 2);
	std::cout << "Minimum number of pens per frame: : " << min_num_pens_ << std::endl;

	nh_.param("bbox_parameters_file", bbox_argus_path_, std::string("/home/robot/dataset_new/bboxArgus.txt"));
	std::cout << "bbox_parameters_file: " << bbox_argus_path_ << std::endl;

	nh_.param("flip_clean_ground", flip_clean_ground_, true);
	std::cout << "flip_clean_ground: " << flip_clean_ground_ << std::endl;

	nh_.param("brightness_shadow_mask_path", brightness_shadow_mask_path_, std::string("/home/rmb-jx/dataset/brightness_shadow_mask"));
	std::cout << "brightness_shadow_mask_path: " << brightness_shadow_mask_path_ << std::endl;

	if (segment_dirt_or_blend_ == 0)
		Segmentation();
	else
		Blending();
}


void ipa_dirt_detection_dataset_tools::DatasetCreate::Segmentation()
{
	SimpleSegmentation segment_dirt(dirts_to_segmented_path_, segmented_dirt_cropped_path_, segmented_dirt_cropped_mask_path_, crop_residual_);
	segment_dirt.run();
}


void ipa_dirt_detection_dataset_tools::DatasetCreate::Blending()
{
	ImageBlender image_blend(ground_image_path_, segmented_dirt_cropped_path_, segmented_dirt_cropped_mask_path_, segmented_pens_path_, segmented_pens_mask_path_,
			blended_ground_image_path_, blended_ground_image_mask_path_, max_num_dirts_, min_num_dirts_, max_num_pens_, min_num_pens_, bbox_argus_path_, flip_clean_ground_,
			brightness_shadow_mask_path_);
	image_blend.run();
}


int main(int argc, char** argv)
{
	ros::init(argc, argv, "dataset_create");
	ros::NodeHandle nh;

	ipa_dirt_detection_dataset_tools::DatasetCreate dc(nh);
	dc.init();

	ros::spin();
	return 1;
}
