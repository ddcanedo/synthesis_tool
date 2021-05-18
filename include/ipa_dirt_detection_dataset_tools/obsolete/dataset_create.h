#ifndef DATASET_CREATE_H
#define DATASET_CREATE_H

#include <ros/ros.h>
#include <iostream>
#include <ipa_dirt_detection_dataset_tools/image_blend.h>
#include <ipa_dirt_detection_dataset_tools/simple_segmentation.h>

namespace ipa_dirt_detection_dataset_tools
{
	class DatasetCreate
	{
	public:
		DatasetCreate(ros::NodeHandle nh);
		void init();
		void Segmentation();
		void Blending();

	protected:
		ros::NodeHandle nh_;

	private:

		bool black_white_background_;              // 0 for black background of the segmented dirt, 1 for white
		bool segment_dirt_or_blend_;               // option for segment dirt or blend the images, 0 for segmentation, 1 for blending

		bool flip_clean_ground_;

		int blended_img_num_;                      // number of blended images to be generated
		int crop_residual_;

		int max_num_dirts_, min_num_dirts_;
		int max_num_pens_, min_num_pens_;

		std::string ground_image_path_;             // The datapath to the clean ground images

		std::string dirts_to_segmented_path_;       // The path to the dirt images which to be segmented

		std::string segmented_dirt_cropped_path_;            // Path to save the segmented dirt after cropping
		std::string segmented_dirt_cropped_mask_path_;       // Path to save the segmented dirt mask after cropping

		std::string segmented_pens_path_;
		std::string segmented_pens_mask_path_;

		std::string blended_ground_image_path_;             // Path to save the blended ground image_blend
		std::string blended_ground_image_mask_path_;        // Path to save the blended ground image mask

		std::string brightness_shadow_mask_path_;

		std::string bbox_argus_path_;
	};
};

#endif
