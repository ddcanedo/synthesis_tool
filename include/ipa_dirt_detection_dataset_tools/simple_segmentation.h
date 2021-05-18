#ifndef SIMPLE_SEGMENT_H
#define SIMPLE_SEGMENT_H

#include <iostream>
#include <string.h>
#include <map>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>
#include <boost/algorithm/string.hpp>
#include <valarray>

namespace ipa_dirt_detection_dataset_tools
{
class SimpleSegmentation
{
public:
	SimpleSegmentation(const std::string src_image_path, const std::string rename_img_to, const std::string cropped_image_path, const std::string cropped_mask_path,
			const double foreground_rectangle_canny1, const double foreground_rectangle_canny2, const double foreground_rectangle_min_area,
			const double foreground_rectangle_target_area, const double foreground_rectangle_shape_threshold, const int foreground_rectangle_additional_cropping,
			const double segmentation_threshold_L_lower, const double segmentation_threshold_L_upper, const double segmentation_threshold_ab, const int crop_residual);
	~SimpleSegmentation();

	void run();

	// find rectangular background
	cv::RotatedRect findRectangularBackground(const cv::Mat& image);

	// crop original image to inner axis aligned rectangle of target rectangular background
	void removeUncontrolledBackground(const cv::Mat& src_image, cv::Mat& cropped_image, const cv::RotatedRect& foreground_rectangle);

	void segment(const cv::Mat& image, cv::Mat& mask_frame);

	void crop(const cv::Mat& dirt_frame, const cv::Mat& mask_frame, cv::Mat& cropped_dirt_frame, cv::Mat& cropped_mask_frame);

	void examine(const cv::Mat& dirt_frame, const cv::Mat& mask_frame);
private:

	// parameters
	std::string source_image_path_;			// path of dirt or object images to be segmented from background - this is the source folder for cropping objects or dirt from images
	std::string rename_img_to_;				// [optional] rename the 'img' part in the file names to this string, the remainder (i.e. the number) stays the same, turned off for empty string ""
	std::string cropped_image_path_;		// path to save the segmented dirt or object after cropping, also used as source for dirt or object samples during image blending
	std::string cropped_mask_path_;			// path to save the cropped dirt or object masks, also used as source for dirt or object masks when blending
	double foreground_rectangle_canny1_;		// Canny edge threshold 1 for finding the foreground rectangle
	double foreground_rectangle_canny2_;		// Canny edge threshold 2 for finding the foreground rectangle
	double foreground_rectangle_min_area_;	// the contour of the foreground rectangle, on which the object or dirt is placed, should fill at least this percentage of the image, in [%] of full image area
	double foreground_rectangle_target_area_;	// the contour of the foreground rectangle, on which the object or dirt is placed, fills approximately this percentage of the image, in [%] of full image area
												// if multiple foreground rectangle areas are found, the one with area closest to this value is preferred
	double foreground_rectangle_shape_threshold_;	// threshold for the similarity of the foreground rectangle to a rectangular shape, in [%] of the area of a perfect rectangle
	int foreground_rectangle_additional_cropping_;	// additional cropping of the identified foreground rectangle from all four sides, in [px]
	double segmentation_threshold_L_lower_;		// segmentation thresholds for luminance difference in the L-channel, in [0,255]
	double segmentation_threshold_L_upper_;		// regions with a threshold between segmentation_threshold_L_lower and segmentation_threshold_L_upper are masked with relative transparency
	double segmentation_threshold_ab_;		// segmentation threshold for a and b difference in the a/b-channels, in [0,255]
	int crop_residual_;			// border residual for cropping bounding box - i.e. crop_residual is the number of pixels added to the mask border to yield the bounding box size
};
};

#endif
