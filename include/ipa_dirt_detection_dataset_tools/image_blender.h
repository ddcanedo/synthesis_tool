#ifndef IMAGE_BLEND_H
#define IMAGE_BLEND_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <ctime>
#include <dirent.h>
#include <math.h>
#include <vector>

// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>
#include <boost/lambda/bind.hpp>
#include <boost/algorithm/string.hpp>

namespace ipa_dirt_detection_dataset_tools
{
class ImageBlender
{
public:
	ImageBlender(const std::string& segmented_marks_path, const std::string& segmented_marks_mask_path, const int max_num_marks, const int min_num_marks, const std::string& clean_ground_path, const std::string& segmented_dirt_path, const std::string& segmented_dirt_mask_path,
			const std::string& segmented_liquids_path, const std::string& segmented_liquids_mask_path, const std::string& brightness_shadow_mask_path,
			const std::string& illumination_mask_path, const std::string& blended_img_folder, const std::string& blended_mask_folder,
			const std::string& blended_img_bbox_filename, const int max_num_dirt, const int min_num_dirt, const int max_num_liquids, const int min_num_liquids,
			const bool flip_clean_ground, const int ground_image_reuse_times);
	~ImageBlender();

	void run();

	// creates lists of all image files (clean images, dirt images and masks, liquid images, illumination and shadow images)
	void collectImageFiles();

	// blend dirt samples into the image
	void blendImageDirt(cv::Mat& blended_image, cv::Mat& blended_mask, cv::Mat& floor_mask, const double clean_ground_image_mean, const int dirt_num, std::vector<cv::Rect>& patch_roi_list,
			std::ofstream& bbox_labels_file, const std::string& base_filename);

	// blend liquid samples into the image
	void blendImageLiquids(cv::Mat& blended_image, cv::Mat& blended_mask, cv::Mat& floor_mask, const double clean_ground_image_mean, const int liquid_num, std::vector<cv::Rect>& patch_roi_list,
			std::ofstream& bbox_labels_file, const std::string& base_filename);
			
	// blend mark samples into the image
	void blendImageMarks(cv::Mat& blended_image, cv::Mat& blended_mask, cv::Mat& floor_mask, const double clean_ground_image_mean, const int mark_num, std::vector<cv::Rect>& patch_roi_list, std::ofstream& bbox_labels_file, const std::string& base_filename);

	// helper function for actually blending the patch into the image with random rotation, scaling, and placement
	void blendImagePatch(cv::Mat& blended_image, cv::Mat& blended_mask, cv::Mat& patch_image, cv::Mat& patch_mask, cv::Mat& floor_mask, const double clean_ground_image_mean,
			std::vector<cv::Rect>& patch_roi_list, std::ofstream& bbox_labels_file, const std::string& base_filename, const std::string& class_name, const int anchor_offset=0);

	// rotates the image and mask
	// @param rotation_angle in [deg]
	void rotateImage(cv::Mat& image, cv::Mat& image_mask, const double rotation_angle, const double scale_factor=1., const int interpolation_mode=cv::INTER_LINEAR);

	// rotates an illumination or shadow mask
	void rotateIlluminationMask(cv::Mat& image, const double rotation_angle, const double scale_factor = 1., const cv::Point translation_offset = cv::Point(0, 0),
			const int interpolation_mode = cv::INTER_LINEAR);

	// reduces an image's bounding box to the area of the mask
	void shrinkBoundingBox(cv::Mat& image, cv::Mat& image_mask);

	// the function is used for extracting the class name from the file name of the segmented patches
	std::string getPatchClassname(const std::string& patch_name);

	void edge_smoothing(cv::Mat& blended_image, cv::Mat& blended_mask, const int half_kernel_size);
	void shadow_and_illuminance(cv::Mat& blended_image, const bool shadow_or_illuminance);
	void addBrightnessOrShadowFromTemplate(cv::Mat& blended_image, const bool add_brightness);
	void addIlluminationFromTemplate(cv::Mat& blended_image);

	void resizeDirt(cv::Mat& dirt_image, cv::Mat& dirt_mask);

private:
	std::vector<std::string> clean_ground_filenames_;
	std::vector<std::string> clean_ground_mask_filenames_;
	std::vector<std::string> segmented_dirt_filenames_;
	std::vector<std::string> segmented_dirt_mask_filenames_;
	std::vector<std::string> segmented_liquids_filenames_;
	std::vector<std::string> segmented_liquids_mask_filenames_;
	std::vector<std::string> segmented_marks_filenames_;
	std::vector<std::string> segmented_marks_mask_filenames_;
	std::vector<std::string> brightness_shadow_mask_filenames_;
	std::vector<std::string> illumination_mask_filenames_;

	int num_clean_ground_images_;
	int num_segmented_dirt_images_;
	int num_segmented_liquid_images_;
	int num_segmented_mark_images_;
	int num_brightness_shadow_mask_images_;
	int num_illumination_mask_images_;


	// parameters
	std::string clean_ground_path_;				// clean ground images path
	std::string segmented_dirt_path_;			// path to the segmented dirt samples
	std::string segmented_dirt_mask_path_;		// path to the segmented dirt masks
	std::string segmented_liquids_path_;		// path to the segmented liquid samples
	std::string segmented_liquids_mask_path_;	// path to the segmented liquid masks
	
	std::string segmented_marks_path_;		// path to the segmented mark samples
	std::string segmented_marks_mask_path_;	// path to the segmented mark masks
	
	std::string brightness_shadow_mask_path_;	// path to brightness and shadow masks, source folder for brightness and shadow masks
	std::string illumination_mask_path_;		// path to illumination masks, source folder for illumination masks

	std::string blended_img_folder_;			// path to save the blended images
	std::string blended_mask_folder_;			// path to save the blended image masks
	std::string blended_img_bbox_filename_;		// name to the output file which stores the parameters of the bounding boxes of the blended images

	int max_num_dirt_;		// maximum number of dirt spots per frame
	int min_num_dirt_;		// minimum number of dirt spots per frame

	int max_num_liquids_;	// maximum number of liquids per frame
	int min_num_liquids_;	// minimum number of liquids per frame
	
	int max_num_marks_;	// maximum number of marks per frame
	int min_num_marks_;	// minimum number of marks per frame

	bool flip_clean_ground_;		// option, whether to flip the clean ground images horizontally and vertically or not (generates 4 images out of one ground image), false=off, true=on
	int ground_image_reuse_times_;	// number of reuses for the same ground pattern (i.e. how many times each image will be used for blending artificial images)
};
template<typename T>
std::string to_string(T Number)
{
	std::ostringstream ss;
	ss << Number;
	return ss.str();
}
}
;

#endif
