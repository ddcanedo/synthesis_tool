#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "ipa_dirt_detection_dataset_tools/image_blender.h"

ipa_dirt_detection_dataset_tools::ImageBlender::ImageBlender(const std::string& segmented_marks_path, const std::string& segmented_marks_mask_path, const int max_num_marks, const int min_num_marks, const std::string& clean_ground_path, const std::string& segmented_dirt_path,
		const std::string& segmented_dirt_mask_path, const std::string& segmented_liquids_path, const std::string& segmented_liquids_mask_path,
		const std::string& brightness_shadow_mask_path, const std::string& illumination_mask_path, const std::string& blended_img_folder,
		const std::string& blended_mask_folder, const std::string& blended_img_bbox_filename, const int max_num_dirt, const int min_num_dirt, const int max_num_liquids,
		const int min_num_liquids, const bool flip_clean_ground, const int ground_image_reuse_times)
{
	clean_ground_path_ = clean_ground_path;

	segmented_dirt_path_ = segmented_dirt_path;
	segmented_dirt_mask_path_ = segmented_dirt_mask_path;
	segmented_liquids_path_ = segmented_liquids_path;
	segmented_liquids_mask_path_ = segmented_liquids_mask_path;

	segmented_marks_path_ = segmented_marks_path;
	segmented_marks_mask_path_ = segmented_marks_mask_path;

	brightness_shadow_mask_path_ = brightness_shadow_mask_path;
	illumination_mask_path_ = illumination_mask_path;

	blended_img_folder_ = blended_img_folder;
	blended_mask_folder_ = blended_mask_folder;
	blended_img_bbox_filename_ = blended_img_bbox_filename;

	max_num_dirt_ = max_num_dirt;
	min_num_dirt_ = min_num_dirt;

	max_num_liquids_ = max_num_liquids;
	min_num_liquids_ = min_num_liquids;

	max_num_marks_ = max_num_marks;
	min_num_marks_ = min_num_marks;

	flip_clean_ground_ = flip_clean_ground;
	ground_image_reuse_times_ = ground_image_reuse_times;
	if (ground_image_reuse_times_ <= 0)
		ground_image_reuse_times_ = 1;


	//srand((int) time(0));
	srand(0);		// keep things repeatable
}

ipa_dirt_detection_dataset_tools::ImageBlender::~ImageBlender()
{
}

void ipa_dirt_detection_dataset_tools::ImageBlender::run()
{
	// create lists of all image files (clean images, dirt images and masks, liquid images, mark images, illumination and shadow images)
	collectImageFiles();

	//-------------------------------------------------------------------------------------------------------------------------------------
	std::ofstream bbox_labels_file;
	//bbox_labels_file.open(blended_img_bbox_filename_.c_str());

	int bg_img_count = 0;

	// start to blend images, reuse the ground pattern with different dirts
	for (int m = 0; m < clean_ground_filenames_.size(); ++m)
	{
		// load clean floor image
		const cv::Mat clean_ground_image = cv::imread(clean_ground_filenames_[m], cv::IMREAD_COLOR);
		cv::Mat clean_ground_image_mask;
		if(std::find(clean_ground_mask_filenames_.begin(), clean_ground_mask_filenames_.end(), clean_ground_filenames_[m].substr(0, clean_ground_filenames_[m].find(".")) + "_mask.png") != clean_ground_mask_filenames_.end())
			clean_ground_image_mask = cv::imread(clean_ground_filenames_[m].substr(0, clean_ground_filenames_[m].find(".")) + "_mask.png", cv::IMREAD_GRAYSCALE);

		// compute average brightness of clean ground image
		const cv::Scalar image_avg = cv::mean(clean_ground_image);
		const double clean_ground_image_mean = (image_avg[0]+image_avg[1]+image_avg[2])/3.;
//		std::cout << "image mean: " << clean_ground_image_mean << std::endl;
//		cv::imshow("clean_ground_image", clean_ground_image);
//		cv::waitKey();

		//-------------------------------------------------------------------------------------------------------
		// create the file name of the blended ground
		std::vector<std::string> strs, strs1;
		boost::split(strs, clean_ground_filenames_[m], boost::is_any_of("\t,/"));			// folders and filename
		boost::split(strs1, strs[strs.size()-1], boost::is_any_of("\t,."));		// remove the suffix of the image name
		const std::string ground_file_name = strs[strs.size()-2] + "_" + strs1[0];
		std::cout << "FILE NAME " << ground_file_name << std::endl;
		//-------------------------------------------------------------------------------------------------------

		// generates as many synthesized images from each clean image as requested by ground_image_reuse_times_
		for (int r=0; r<ground_image_reuse_times_; ++r)
		{
			// apply image flipping if requested by flip_clean_ground_
			const int max_flipping_operations = (flip_clean_ground_==true) ? 4 : 1;
			for (int flip_index=0; flip_index<max_flipping_operations; ++flip_index)
			{
				// apply flip operation
				cv::Mat blended_image = clean_ground_image.clone();
				cv::Mat floor_mask = clean_ground_image_mask.clone();
				if (flip_index == 1){
					cv::flip(clean_ground_image, blended_image, 0);
					cv::flip(clean_ground_image_mask, floor_mask, 0);
				}
				if (flip_index == 2){
					cv::flip(clean_ground_image, blended_image, 1);
					cv::flip(clean_ground_image_mask, floor_mask, 1);
				}
				if (flip_index == 3){
					cv::flip(clean_ground_image, blended_image, -1);
					cv::flip(clean_ground_image_mask, floor_mask, -1);
				}

				const std::string base_filename(ground_file_name + "_r" + ipa_dirt_detection_dataset_tools::to_string(r) + "_f" + ipa_dirt_detection_dataset_tools::to_string(flip_index));
				
				// generate some background images (1%)
				if(rand()%100 == 0){
					bg_img_count++;
					cv::imwrite(blended_img_folder_ + "/images/train/" + base_filename + ".png", blended_image);
					continue;
				}
				// create mask
				cv::Mat blended_mask = cv::Mat::zeros(blended_image.rows, blended_image.cols, CV_8UC1);

				// random numbers of dirt, liquids and marks
				const int dirt_num = min_num_dirt_ + rand() % (max_num_dirt_ - min_num_dirt_);		// generate number of dirts in range max_num_dirt and min_num_dirt
				const int liquids_num = min_num_liquids_ + rand() % (max_num_liquids_ - min_num_liquids_);		// generate number of liquids in range max_num_liquids and min_num_liquids
				const int marks_num = min_num_marks_ + rand() % (max_num_marks_ - min_num_marks_);		// generate number of marks in range max_num_marks and min_num_marks

				// blend images
				std::vector<cv::Rect> patch_roi_list;		// keeps track of utilized ROIs for placing patches
				bbox_labels_file.open(blended_img_folder_ + "/labels/train/" + base_filename + ".txt");
				blendImageMarks(blended_image, blended_mask, floor_mask, clean_ground_image_mean, marks_num, patch_roi_list, bbox_labels_file, base_filename);	// first add the marks
				blendImageDirt(blended_image, blended_mask, floor_mask, clean_ground_image_mean, dirt_num, patch_roi_list, bbox_labels_file, base_filename);			// then add artificial dirt,
				blendImageLiquids(blended_image, blended_mask, floor_mask, clean_ground_image_mean, liquids_num, patch_roi_list, bbox_labels_file, base_filename);	// then add the liquids

				// add illumination for 25% of the images
				if (rand()%4 == 0)
					addIlluminationFromTemplate(blended_image);

				// add brightness and shadows for 25% of the images
				if (rand()%10 == 0)
					addBrightnessOrShadowFromTemplate(blended_image, false);	// shadows

				if (rand()%7 == 0)
					addBrightnessOrShadowFromTemplate(blended_image, true);		// brightness
				//edge_smoothing(3);
				// store blended images and masks
				//cv::imwrite(blended_mask_folder_ + "/" + base_filename + "_mask.png", blended_mask);
				//cv::imwrite(blended_mask_folder_ + "/" + base_filename + "_mask_vis.png", blended_mask * 255);
				cv::imwrite(blended_img_folder_ + "/images/train/" + base_filename + ".png", blended_image);
				bbox_labels_file.close();
			}
		}
		std::cout << "Background images: " << bg_img_count << std::endl;
	}
}


void ipa_dirt_detection_dataset_tools::ImageBlender::collectImageFiles()
{
	boost::filesystem::directory_iterator end_itr;

	// to list the clean ground file names
	for (boost::filesystem::directory_iterator itr1(clean_ground_path_); itr1 != end_itr; ++itr1)
	{
		for (boost::filesystem::directory_iterator itr2(itr1->path()); itr2 != end_itr; ++itr2)
			if (itr2->path().string().find("_mask.png") == std::string::npos)
				clean_ground_filenames_.push_back(itr2->path().string());
			else
				clean_ground_mask_filenames_.push_back(itr2->path().string());
	}
	num_clean_ground_images_ = clean_ground_filenames_.size();
	std::cout << "The number of clean ground images is " << num_clean_ground_images_ << std::endl;
	std::cout << "The number of clean ground masks is " << clean_ground_mask_filenames_.size() << std::endl;
	std::sort(clean_ground_filenames_.begin(), clean_ground_filenames_.end());
	for (std::vector<std::string>::iterator it = clean_ground_filenames_.begin(); it != clean_ground_filenames_.end(); ++it)
		std::cout << "   - " << *it << std::endl;

	// to list the segmented dirt file names
	for (boost::filesystem::directory_iterator itr2(segmented_dirt_path_); itr2 != end_itr; ++itr2)
		if (itr2->path().string().find("_mask.png") == std::string::npos)
			segmented_dirt_filenames_.push_back(itr2->path().string());
	num_segmented_dirt_images_ = segmented_dirt_filenames_.size();
	std::cout << "\nThe number of dirt samples is " << num_segmented_dirt_images_ << std::endl;
	std::sort(segmented_dirt_filenames_.begin(), segmented_dirt_filenames_.end());
	for (std::vector<std::string>::iterator it = segmented_dirt_filenames_.begin(); it != segmented_dirt_filenames_.end(); ++it){
		std::cout << "   - " << *it << std::endl;
		std::string aux = *it;
		segmented_dirt_mask_filenames_.push_back(aux.substr(0, aux.find(".")) + "_mask.png");
	}

	// to list the segmented dirt mask file names
	//for (boost::filesystem::directory_iterator itr3(segmented_dirt_mask_path_); itr3 != end_itr; ++itr3)
	//	if (itr3->path().string().find("_mask.png") != std::string::npos)
	//		segmented_dirt_mask_filenames_.push_back(itr3->path().string());
	//std::cout << "\nThe number of dirt masks is " << segmented_dirt_mask_filenames_.size() << std::endl;
	//std::sort(segmented_dirt_mask_filenames_.begin(), segmented_dirt_mask_filenames_.end());
	for (std::vector<std::string>::iterator it = segmented_dirt_mask_filenames_.begin(); it != segmented_dirt_mask_filenames_.end(); ++it)
		std::cout << "   - " << *it << std::endl;

	// to list the liquids file names
	for (boost::filesystem::directory_iterator itr4(segmented_liquids_path_); itr4 != end_itr; ++itr4)
		if (itr4->path().string().find("_mask.png") == std::string::npos)
			segmented_liquids_filenames_.push_back(itr4->path().string());
	num_segmented_liquid_images_ = segmented_liquids_filenames_.size();
	std::cout << "\nThe number of liquid samples is " << num_segmented_liquid_images_ << std::endl;
	std::sort(segmented_liquids_filenames_.begin(), segmented_liquids_filenames_.end());
	for (std::vector<std::string>::iterator it = segmented_liquids_filenames_.begin(); it != segmented_liquids_filenames_.end(); ++it){
		std::cout << "   - " << *it << std::endl;
		std::string aux = *it;
		segmented_liquids_mask_filenames_.push_back(aux.substr(0, aux.find(".")) + "_mask.png");
	}

	// to list the liquid mask file names
	//for (boost::filesystem::directory_iterator itr5(segmented_liquids_mask_path_); itr5 != end_itr; ++itr5)
	//	if (itr5->path().string().find("_mask.png") != std::string::npos)
	//		segmented_liquids_mask_filenames_.push_back(itr5->path().string());
	//std::cout << "\nThe number of liquid masks is " << segmented_liquids_mask_filenames_.size() << std::endl;
	//std::sort(segmented_liquids_mask_filenames_.begin(), segmented_liquids_mask_filenames_.end());
	for (std::vector<std::string>::iterator it = segmented_liquids_mask_filenames_.begin(); it != segmented_liquids_mask_filenames_.end(); ++it)
		std::cout << "   - " << *it << std::endl;



	// to list the marks file names
	for (boost::filesystem::directory_iterator itr5(segmented_marks_path_); itr5 != end_itr; ++itr5)
		if (itr5->path().string().find("_mask.png") == std::string::npos)
			segmented_marks_filenames_.push_back(itr5->path().string());
	num_segmented_mark_images_ = segmented_marks_filenames_.size();
	std::cout << "\nThe number of marks samples is " << num_segmented_mark_images_ << std::endl;
	std::sort(segmented_marks_filenames_.begin(), segmented_marks_filenames_.end());
	for (std::vector<std::string>::iterator it = segmented_marks_filenames_.begin(); it != segmented_marks_filenames_.end(); ++it){
		std::cout << "   - " << *it << std::endl;
		std::string aux = *it;
		segmented_marks_mask_filenames_.push_back(aux.substr(0, aux.find(".")) + "_mask.png");
	}

	
	for (std::vector<std::string>::iterator it = segmented_marks_mask_filenames_.begin(); it != segmented_marks_mask_filenames_.end(); ++it)
		std::cout << "   - " << *it << std::endl;



	// to list the brightness and shadow masks
	for (boost::filesystem::directory_iterator itr6(brightness_shadow_mask_path_); itr6 != end_itr; ++itr6)
		brightness_shadow_mask_filenames_.push_back(itr6->path().string());
	num_brightness_shadow_mask_images_ = brightness_shadow_mask_filenames_.size();
	std::cout << "\nThe number of brightness and shadow masks is " << num_brightness_shadow_mask_images_ << std::endl;
	std::sort(brightness_shadow_mask_filenames_.begin(), brightness_shadow_mask_filenames_.end());
	for (std::vector<std::string>::iterator it = brightness_shadow_mask_filenames_.begin(); it != brightness_shadow_mask_filenames_.end(); ++it)
		std::cout << "   - " << *it << std::endl;

	// to list the illumination masks
	for (boost::filesystem::directory_iterator itr7(illumination_mask_path_); itr7 != end_itr; ++itr7)
		illumination_mask_filenames_.push_back(itr7->path().string());
	num_illumination_mask_images_ = illumination_mask_filenames_.size();
	std::cout << "\nThe number of illumination masks is " << num_illumination_mask_images_ << std::endl;
	std::sort(illumination_mask_filenames_.begin(), illumination_mask_filenames_.end());
	for (std::vector<std::string>::iterator it = illumination_mask_filenames_.begin(); it != illumination_mask_filenames_.end(); ++it)
		std::cout << "   - " << *it << std::endl;
}


void ipa_dirt_detection_dataset_tools::ImageBlender::blendImageDirt(cv::Mat& blended_image, cv::Mat& blended_mask, cv::Mat& floor_mask, const double clean_ground_image_mean, const int dirt_num,
		std::vector<cv::Rect>& patch_roi_list, std::ofstream& bbox_labels_file, const std::string& base_filename)
{
	for (int n = 0; n < dirt_num; ++n)
	{
		// select and load dirt image and mask
		const int dirt_image_index = rand() % num_segmented_dirt_images_;
		cv::Mat dirt_image = cv::imread(segmented_dirt_filenames_[dirt_image_index], cv::IMREAD_COLOR);
		//std::cout << segmented_dirt_filenames_[dirt_image_index] << std::endl;
		cv::Mat dirt_mask = cv::imread(segmented_dirt_mask_filenames_[dirt_image_index], cv::IMREAD_GRAYSCALE);
		//std::cout << segmented_dirt_mask_filenames_[dirt_image_index] << std::endl;

		blendImagePatch(blended_image, blended_mask, dirt_image, dirt_mask, floor_mask, clean_ground_image_mean, patch_roi_list, bbox_labels_file, base_filename, "dirt", 0);
		//std::cout << "------- one dirt finished" << std::endl;
	}
}

void ipa_dirt_detection_dataset_tools::ImageBlender::blendImageLiquids(cv::Mat& blended_image, cv::Mat& blended_mask, cv::Mat& floor_mask, const double clean_ground_image_mean,
		const int liquid_num, std::vector<cv::Rect>& patch_roi_list, std::ofstream& bbox_labels_file, const std::string& base_filename)
{
	for (int n = 0; n < liquid_num; ++n)
	{
		// select and load liquid image and mask
		const int liquid_image_index = rand() % num_segmented_liquid_images_;
		cv::Mat liquid_image = cv::imread(segmented_liquids_filenames_[liquid_image_index], cv::IMREAD_COLOR);
		//std::cout << segmented_liquids_filenames_[liquid_image_index] << std::endl;
		cv::Mat liquid_mask = cv::imread(segmented_liquids_mask_filenames_[liquid_image_index], cv::IMREAD_GRAYSCALE);
		//std::cout << segmented_liquids_mask_filenames_[liquid_image_index] << std::endl;

		blendImagePatch(blended_image, blended_mask, liquid_image, liquid_mask, floor_mask, clean_ground_image_mean, patch_roi_list, bbox_labels_file, base_filename, "liquid", 0);
		//std::cout << "------- one liquid finished" << std::endl;
	}
}



void ipa_dirt_detection_dataset_tools::ImageBlender::blendImageMarks(cv::Mat& blended_image, cv::Mat& blended_mask, cv::Mat& floor_mask, const double clean_ground_image_mean,
		const int mark_num, std::vector<cv::Rect>& patch_roi_list, std::ofstream& bbox_labels_file, const std::string& base_filename)
{
	for (int n = 0; n < mark_num; ++n)
	{
		// select and load mark image and mask
		const int mark_image_index = rand() % num_segmented_mark_images_;
		cv::Mat mark_image = cv::imread(segmented_marks_filenames_[mark_image_index], cv::IMREAD_COLOR);
		//std::cout << segmented_marks_filenames_[mark_image_index] << std::endl;
		cv::Mat mark_mask = cv::imread(segmented_marks_mask_filenames_[mark_image_index], cv::IMREAD_GRAYSCALE);
		//std::cout << segmented_marks_mask_filenames_[mark_image_index] << std::endl;
		blendImagePatch(blended_image, blended_mask, mark_image, mark_mask, floor_mask, clean_ground_image_mean, patch_roi_list, bbox_labels_file, base_filename, "mark", 0);
		//std::cout << "------- one mark finished" << std::endl;
	}
}



void ipa_dirt_detection_dataset_tools::ImageBlender::blendImagePatch(cv::Mat& blended_image, cv::Mat& blended_mask, cv::Mat& patch_image, cv::Mat& patch_mask, cv::Mat& floor_mask,
		const double clean_ground_image_mean, std::vector<cv::Rect>& patch_roi_list, std::ofstream& bbox_labels_file, const std::string& base_filename,
		const std::string& class_name, const int anchor_offset)
{
	// adapt sample on dark images
	if (clean_ground_image_mean < 64.)
		patch_image *= std::max(0.33, clean_ground_image_mean/64.);

//	// remove some edges
//	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3), cv::Point(1, 1));
//	cv::morphologyEx(patch_mask, patch_mask, cv::MORPH_ERODE, element);

	// 20% of the dirt will be resized
	double scale_factor = 1.;
	int interpolation_mode = cv::INTER_LINEAR;
	double alpha = 1.;
	double rotation_angle = 0;
	double floor_min_side = 0;
	double side_ratio = 0;
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
	//if (class_name.compare("dirt")==0 && rand()%5 == 0)
	//{
		//scale_factor = (rand() % 41 + 80) * 0.01;		// resize ratio in range 0.8 to 1.2


	double dirt_max_side = std::max(patch_image.cols,patch_image.rows);
	if(class_name == "mark")
	{
		alpha = (35 + ( rand() % ( 50 - 35 + 1 ) ))*0.01; // the idea is that dirt is rescaled in relation to image size
		rotation_angle = (rand() % 4)*90;
	}	
	else
	{
		alpha = (5 + ( rand() % ( 7 - 5 + 1 ) ))*0.01; // the idea is that dirt is rescaled in relation to image size
		rotation_angle = rand() % 360;	
	}


	if(floor_mask.empty() == true)
	{
		floor_min_side = std::min(blended_image.cols, blended_image.rows);
		side_ratio = dirt_max_side/(floor_min_side*alpha);
		scale_factor = 1/side_ratio;
	}
	else
	{
		cv::findContours(floor_mask, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
		cv::Rect boundRect = boundingRect( contours[0] ); // using the rect to calculate the perimeter since arcLength() was not working properly
		double floor_perimeter = 2*(boundRect.width + boundRect.height);
		floor_min_side = std::min(boundRect.width, boundRect.height);
		for( int i = 1; i< contours.size(); i++ )
		{
			boundRect = boundingRect( contours[i] );
			double current_perimeter = 2*(boundRect.width + boundRect.height);
			if(current_perimeter > floor_perimeter){
				floor_perimeter = current_perimeter;
				floor_min_side = std::min(boundRect.width, boundRect.height);
			}
		}
		side_ratio = dirt_max_side/(floor_min_side*alpha);
		scale_factor = 1/side_ratio;
	}

	//std::cout << "resize scale_factor: " << scale_factor << std::endl;
	if (scale_factor < 1.)
		interpolation_mode = cv::INTER_AREA;		// best for shrinking images
	if (scale_factor > 1.)
		interpolation_mode = cv::INTER_CUBIC;	// best for enlarging images
	//}


	//std::cout << "The rotation angle is " << rotation_angle << std::endl;
	rotateImage(patch_image, patch_mask, rotation_angle, scale_factor, interpolation_mode);
	shrinkBoundingBox(patch_image, patch_mask);

	// place image patch at random location but avoid larger overlap with already blended items
	patch_image.convertTo(patch_image, blended_image.type());
	const int patch_cols = patch_image.cols;
	const int patch_rows = patch_image.rows;
	int anchor_col = 0, anchor_row = 0;
	bool is_within_mask = true;
	bool has_significant_overlap = false;
	for (int k=0; k<1000; ++k)
	{
		if(k == 999)
			std::cout << "Blending failed" << std::endl;
		anchor_col = anchor_offset + rand() % blended_image.cols;		// top left point of the blending position
		anchor_row = anchor_offset + rand() % blended_image.rows;
		if ((anchor_col + patch_cols + anchor_offset) >= blended_image.cols)
			anchor_col = blended_image.cols - 1 - (patch_cols + anchor_offset);
		if ((anchor_row + patch_rows + anchor_offset) >= blended_image.rows)
			anchor_row = blended_image.rows - 1 - (patch_rows + anchor_offset);

		is_within_mask = true;
		if(floor_mask.empty() == false){
			for (int v=anchor_row - anchor_offset; v<anchor_row + patch_rows + anchor_offset; ++v)
			{
				for (int u=anchor_col - anchor_offset; u<anchor_col + patch_cols + anchor_offset; ++u)
				{	
					if ((double)floor_mask.at<uchar>(v,u) == 0)
					{
						is_within_mask = false;
						break;
					}
				}
				if(is_within_mask == false)
					break;
			}
		}
		if(is_within_mask == false)
			continue;

		const cv::Rect roi(anchor_col-anchor_offset, anchor_row-anchor_offset, patch_cols+2*anchor_offset, patch_rows+2*anchor_offset);
		has_significant_overlap = false;
		for (size_t l=0; l<patch_roi_list.size(); ++l)
		{
			const double area_overlap = (patch_roi_list[l] & roi).area();
			if (area_overlap > 0.25*patch_roi_list[l].area() || area_overlap > 0.25*roi.area())		// todo: param for 0.25
			{
				has_significant_overlap = true;
				break;
			}
		}

		if (has_significant_overlap == false)
		{
			patch_roi_list.push_back(roi);
			break;
		}
	}

	// write the arguments of the bounding box in file
	//bbox_labels_file << base_filename << " " << anchor_col - anchor_offset << " " << anchor_row - anchor_offset << " "
	//		<< anchor_col + patch_cols + anchor_offset << " " << anchor_row + patch_rows + anchor_offset << " " << class_name << " \n";
	if(is_within_mask == true || has_significant_overlap == false)
	{
		double aux1 = (double)patch_cols/2;
		double aux2 = (double)patch_rows/2;
		double center_x = (double)((anchor_col - anchor_offset) + aux1)/blended_image.cols;
		double center_y = (double)((anchor_row - anchor_offset) + aux2)/blended_image.rows;
		double yolo_width = (double)patch_cols/blended_image.cols;
		double yolo_height = (double)patch_rows/blended_image.rows;
		bbox_labels_file << "0" << " " << center_x << " " << center_y << " " << yolo_width << " " << yolo_height << "\n";

		// modify borders of the mask for smooth transition to background
		cv::Mat patch_mask_thresholded, patch_mask_eroded;
		cv::threshold(patch_mask, patch_mask_thresholded, 0, 255, cv::THRESH_BINARY);
		cv::erode(patch_mask_thresholded, patch_mask_eroded, cv::Mat());
		patch_mask_eroded = patch_mask_thresholded - patch_mask_eroded;
		for (int v=0; v<patch_mask.rows; ++v)
			for (int u=0; u<patch_mask.cols; ++u)
				if (patch_mask_eroded.at<uchar>(v,u) != 0)
					patch_mask.at<uchar>(v,u) = 0.5*patch_mask.at<uchar>(v,u);

		// blend patch into the image
		double liquid_opacity = (0 + ( rand() % ( 6 - 0 + 1 ) ))*0.1;
		double mark_opacity = (0 + ( rand() % ( 5 - 0 + 1 ) ))*0.1;
		for (int i = anchor_row; i < anchor_row + patch_rows; ++i)
		{
			for (int j = anchor_col; j < anchor_col + patch_cols; ++j)
			{
				const int mask_row = i - anchor_row;
				const int mask_col = j - anchor_col;
				const double opacity_factor = (double)patch_mask.at<uchar>(mask_row, mask_col) * 1./255.;
				if (opacity_factor > 0.)
				{
					/*if (opacity_factor > 0.999)
						blended_image.at<cv::Vec3b>(i, j) = patch_image.at<cv::Vec3b>(mask_row, mask_col);
					else
						blended_image.at<cv::Vec3b>(i, j) += opacity_factor * (patch_image.at<cv::Vec3b>(mask_row, mask_col)-blended_image.at<cv::Vec3b>(i, j));
					blended_mask.at<uchar>(i, j) = 1;*/
					if (class_name == "liquid"){
						cv::addWeighted(blended_image.at<cv::Vec3b>(i, j), liquid_opacity, patch_image.at<cv::Vec3b>(mask_row, mask_col), 1.0-liquid_opacity, 0.0, blended_image.at<cv::Vec3b>(i, j));
					}
					else if(class_name == "mark"){
						cv::addWeighted(blended_image.at<cv::Vec3b>(i, j), mark_opacity, patch_image.at<cv::Vec3b>(mask_row, mask_col), 1.0-mark_opacity, 0.0, blended_image.at<cv::Vec3b>(i, j));
					}
					else{
						if (opacity_factor > 0.999)
							blended_image.at<cv::Vec3b>(i, j) = patch_image.at<cv::Vec3b>(mask_row, mask_col);
						else
							blended_image.at<cv::Vec3b>(i, j) += opacity_factor * (patch_image.at<cv::Vec3b>(mask_row, mask_col)-blended_image.at<cv::Vec3b>(i, j));
					}
					blended_mask.at<uchar>(i, j) = 1;
				}
			}
		}
	}
//	cv::Point pt1(anchor_col - anchor_offset, anchor_row - anchor_offset);
//	cv::Point pt2(anchor_col + patch_cols + anchor_offset, anchor_row + patch_rows + anchor_offset);

//	cv::rectangle(blended_image, pt1,pt2, 1, cv::LINE_4);
//	cv::imshow("orig", blended_image);
//	cv::waitKey();
}

// reference: https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
void ipa_dirt_detection_dataset_tools::ImageBlender::rotateImage(cv::Mat& image, cv::Mat& mask, const double rotation_angle, const double scale_factor, const int interpolation_mode)
{
	cv::Point2f rotation_center = cv::Point(ceil(image.cols / 2), ceil(image.rows / 2));
	cv::Mat rotation_matrix = cv::getRotationMatrix2D(rotation_center, rotation_angle, scale_factor);
//	std::cout << rotation_matrix.at<double>(0, 0) << rotation_matrix.at<double>(0, 1) << std::endl;

	// resize target image to contain all rotated data
	const double c = fabs(rotation_matrix.at<double>(0, 0));
	const double s = fabs(rotation_matrix.at<double>(0, 1));
	const int new_width = image.rows * s + image.cols * c;
	const int new_height = image.rows * c + image.cols * s;
	rotation_matrix.at<double>(0, 2) += (new_width / 2) - (image.cols / 2);
	rotation_matrix.at<double>(1, 2) += (new_height / 2) - (image.rows / 2);

//	std::cout << "start to rotate matrix" << std::endl;
//	std::cout << image.cols << ' ' << image.rows << ' ' << new_width << ' ' << new_height << std::endl;

	if (image.type() != CV_8UC3)
		std::cout << "ImageBlender::rotateImage: Error: The image format of the color image is not CV_8UC3.\n" << std::endl;

//	cv::imshow("orig", image);

	cv::Mat temp, temp2;
	cv::warpAffine(image, temp, rotation_matrix, cv::Size(new_width, new_height), interpolation_mode);
	image = temp;
	cv::warpAffine(mask, temp2, rotation_matrix, cv::Size(new_width, new_height), interpolation_mode);
	mask = temp2;

//	cv::imshow("rot", image);
//	cv::imshow("rot_mask", mask);
//	cv::waitKey();
}

void ipa_dirt_detection_dataset_tools::ImageBlender::rotateIlluminationMask(cv::Mat& image, const double rotation_angle, const double scale_factor, const cv::Point translation_offset,
		const int interpolation_mode)
{
	cv::Point2f rotation_center = cv::Point(ceil(image.cols / 2), ceil(image.rows / 2));
	cv::Mat rotation_matrix = cv::getRotationMatrix2D(rotation_center, rotation_angle, scale_factor);
	rotation_matrix.at<double>(0, 2) += translation_offset.x;
	rotation_matrix.at<double>(1, 2) += translation_offset.y;

//	std::cout << "rotation_angle=" << rotation_angle << "    scale_factor=" << scale_factor << "   translation_offset=(" << translation_offset.x << "," << translation_offset.y << ")" << std::endl;
//	cv::imshow("orig", image);

	cv::Mat temp;
	cv::warpAffine(image, temp, rotation_matrix, cv::Size(image.cols, image.rows), interpolation_mode);
	image = temp;

//	cv::imshow("rot", image);
//	cv::waitKey();
}


void ipa_dirt_detection_dataset_tools::ImageBlender::shrinkBoundingBox(cv::Mat& image, cv::Mat& image_mask)
{
	int left_edge = image_mask.cols-1;
	int right_edge = 0;
	int top_edge = image_mask.rows-1;
	int bottom_edge = 0;

	bool mask_empty = true;
	for (int v=0; v<image_mask.rows; ++v)
	{
		for (int u=0; u<image_mask.cols; ++u)
		{
			if (image_mask.at<uchar>(v,u) != 0)
			{
				left_edge = std::min(left_edge, u);
				right_edge = std::max(right_edge, u);
				top_edge = std::min(top_edge, v);
				bottom_edge = std::max(bottom_edge, v);
				mask_empty = false;
			}
		}
	}

	const int mask_padding = 2;
	cv::Rect roi;
	if (mask_empty == false)
	{
		const int max_x = std::min(image.cols-1, right_edge+mask_padding+1);
		const int max_y = std::min(image.rows-1, bottom_edge+mask_padding+1);
		roi.x = std::max(0, left_edge-mask_padding);
		roi.y = std::max(0, top_edge-mask_padding);
		roi.width = max_x - roi.x;
		roi.height = max_y - roi.y;
	}
	else
	{
		roi.x = 0;
		roi.y = 0;
		roi.width = image.cols-1;
		roi.height = image.rows-1;
	}

	cv::Mat temp = image(roi);
	image = temp;
	cv::Mat temp2 = image_mask(roi);
	image_mask = temp2;

//	// For TEST
//	cv::imwrite("/home/robot/artificial dirt1.jpg", artificial_dirt_);
//	cv::imwrite("/home/robot/rotate dirt1.jpg", rotated_artificial_dirt_);
//	cv::imwrite("/home/robot/schrank dirt1.jpg", rotated_schranked_artificial_dirt_);

//	cv::imshow("shrinked_image", image);
//	cv::waitKey();
}

// the function is used for extracting the class name from the file name of the segmented patches
std::string ipa_dirt_detection_dataset_tools::ImageBlender::getPatchClassname(const std::string& patch_name)
{
	// single out image file name
	std::vector<std::string> strs;
	boost::split(strs, patch_name, boost::is_any_of("\t,/"));
	std::string image_name = strs[strs.size() - 1];

	// removed digits after last underscore _
	std::vector<std::string> splits;
	boost::split(splits, image_name, boost::is_any_of("\t,_"));
	std::string liquid_class = splits[0];
	for (size_t k=1; k<splits.size()-1; ++k)
		liquid_class += "_" + splits[k];
	//std::cout << "class of the liquid is: " << splits[0] << std::endl;
	return liquid_class;
}

void ipa_dirt_detection_dataset_tools::ImageBlender::edge_smoothing(cv::Mat& blended_image, cv::Mat& blended_mask, const int half_kernel_size)
{
	// modify borders of the mask for smooth transition to background   --->   the effect is not so great
	cv::Mat blended_mask_eroded, blended_mask_dilated, blended_image_blurred, mask_temp;
	mask_temp = blended_mask*255;
	cv::erode(mask_temp, blended_mask_eroded, cv::Mat());
	cv::dilate(mask_temp, blended_mask_dilated, cv::Mat());
	blended_mask_dilated = blended_mask_dilated - blended_mask_eroded;
	cv::GaussianBlur(blended_image, blended_image_blurred, cv::Size(3,3), 0);
	for (int v=0; v<blended_image.rows; ++v)
		for (int u=0; u<blended_image.cols; ++u)
			if (blended_mask_dilated.at<uchar>(v,u) != 0)
				blended_image.at<cv::Vec3b>(v,u) = blended_image_blurred.at<cv::Vec3b>(v,u);

	return;

	// old code from here:

	blended_image.convertTo(blended_image, CV_32F);

	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;

	// each contour is stored as a vector of points, than we only need to loop over these points to smooth the liquid edge
	cv::Mat temp = blended_mask.clone();
	cv::findContours(temp, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

	/*
	 cv::Mat drawing = blended_image.clone();
	 for( int i = 0; i< contours.size(); i++ )
	 {
	 cv::Scalar color = cv::Scalar(255,0,0);
	 cv::drawContours( drawing , contours, i, color, 2, 8, hierarchy, 0, Point() );
	 }
	 drawing = drawing / 255;
	 cv::imshow("drawing", drawing);
	 cv::waitKey(0);
	 */

	// gaussian blur kernel
	cv::Mat gaussian_x = cv::getGaussianKernel(half_kernel_size * 2 + 1, 0.5, CV_32F);   // second parameter is sigma
	cv::Mat gaussian_y = cv::getGaussianKernel(half_kernel_size * 2 + 1, 0.5, CV_32F);
	cv::Mat gaussian_kernel = gaussian_x * gaussian_y.t();
	std::cout << "Gaussian kernel is: " << gaussian_kernel << std::endl;

	// loop over the contour points for edge_smoothing
	int num_contours = contours.size();
	for (int contour_iter = 0; contour_iter < num_contours; contour_iter++)
	{
		std::vector<cv::Point> current_contour = contours[contour_iter];
		int num_points = current_contour.size();

		for (int point_iter = 0; point_iter < num_points; point_iter++)
		{
			cv::Point anchor_point = current_contour[point_iter];
			int point_x = anchor_point.x;
			int point_y = anchor_point.y;

			if ((point_x - half_kernel_size) < 0 || (point_x + half_kernel_size) >= blended_image.cols || (point_y - half_kernel_size) < 0 || (point_y + half_kernel_size) >= blended_image.rows) // if the ROI is out of the image range, continue
				continue;

			cv::Mat image_roi;
			cv::Rect region_of_interest(point_x - half_kernel_size, point_y - half_kernel_size, 2 * half_kernel_size + 1, 2 * half_kernel_size + 1);
			image_roi = blended_image(region_of_interest);

			// split three color channels
			std::vector<cv::Mat> rgb;
			cv::split(image_roi, rgb);
			cv::Mat b = rgb[0];
			float new_pixel_value_b = cv::sum(b.mul(gaussian_kernel))[0];
			cv::Mat g = rgb[1];
			float new_pixel_value_g = cv::sum(g.mul(gaussian_kernel))[0];
			cv::Mat r = rgb[2];
			float new_pixel_value_r = cv::sum(r.mul(gaussian_kernel))[0];

			blended_image.at<cv::Vec3f>(point_y, point_x)[0] = new_pixel_value_b;      // check point_x and point_y order TODO
			blended_image.at<cv::Vec3f>(point_y, point_x)[1] = new_pixel_value_g;
			blended_image.at<cv::Vec3f>(point_y, point_x)[2] = new_pixel_value_r;

		}
	}
}

// The function used to add artificial shadows and brightness in synthetic images
void ipa_dirt_detection_dataset_tools::ImageBlender::shadow_and_illuminance(cv::Mat& blended_image, const bool shadow_or_illuminance)
{
	// random generator for the shape and size
	int polygon_or_ellipse = rand() % 2;           // 1 for polygon and 0 for ellipse
	cv::Mat mask = cv::Mat::zeros(blended_image.rows, blended_image.cols, CV_32F);
	float f = (rand() % 5) / 10;           //  opacity factor as in Gimp

	if (polygon_or_ellipse)                        // assume the polygon with 4 points
	{
		cv::Point rook_points[1][4];
		int start_point_x = rand() % blended_image.cols;
		int start_point_y = rand() % blended_image.rows;
		float x1 = rand() % abs(start_point_x - 80) + 40;
		float y1 = rand() % abs(start_point_y - 80) + 40;
		float x2 = rand() % abs(start_point_x - 80) + 40;
		float y2 = rand() % abs(abs(start_point_y - blended_image.rows) - 80) + 40;
		float x3 = rand() % abs(abs(start_point_x - blended_image.cols) - 80) + 40;
		float y3 = rand() % abs(abs(start_point_y - blended_image.rows) - 80) + 40;
		float x4 = rand() % abs(abs(start_point_x - blended_image.cols) - 80) + 40;
		float y4 = rand() % abs(start_point_y - 80) + 40;
		// minimum coordinate distance between each point with the center is 80, maximun is 40

		rook_points[0][0] = cv::Point(x1, y1);          // start points
		rook_points[0][1] = cv::Point(x2, y2);
		rook_points[0][2] = cv::Point(x3, y3);
		rook_points[0][3] = cv::Point(x4, y4);

		const cv::Point* ppt[1] =
		{ rook_points[0] };
		int npt[] =
		{ 4 };

		cv::fillPoly(mask, ppt, npt, 1, cv::Scalar(255, 255, 255), 8);
	}
	else
	{
		int start_x = rand() % blended_image.cols;
		int start_y = rand() % blended_image.rows;        // the minimum length of the axis is 40, maximun is 80
		int axes_x = rand() % 80 + 60;
		int axes_y = rand() % 80 + 60;              // TODO check if out of image range cause any problems

		int angle = rand() % 360;
		int start_angle = rand() % 90;
		int end_angle = rand() % 180 + 180;

		cv::ellipse(mask, cv::Point(start_x, start_y), cv::Size(axes_x, axes_y), angle, start_angle, end_angle, cv::Scalar(255, 255, 255), -1, 8);
	}

	if (shadow_or_illuminance)      // 1 for shadow
	{
		mask.convertTo(mask, blended_image.type());
		cv::GaussianBlur(mask, mask, cv::Size(31, 31), 50, 50);

		std::vector<cv::Mat> rgb;
		cv::split(blended_image, rgb);
		rgb[0] = rgb[0] - f * mask;
		rgb[1] = rgb[1] - f * mask;
		rgb[2] = rgb[2] - f * mask;
		cv::merge(rgb, blended_image);
	}
	else
	{
		mask.convertTo(mask, blended_image.type());
		cv::GaussianBlur(mask, mask, cv::Size((blended_image.cols / 3 / 2) * 2 + 1, (blended_image.cols / 3 / 2) * 2 + 1), 50, 50);
		std::vector<cv::Mat> rgb;

		cv::split(blended_image, rgb);
		rgb[0] = rgb[0] + f * mask;
		rgb[1] = rgb[1] + f * mask;
		rgb[2] = rgb[2] + f * mask;
		cv::merge(rgb, blended_image);
	}
}

void ipa_dirt_detection_dataset_tools::ImageBlender::addBrightnessOrShadowFromTemplate(cv::Mat& blended_image, const bool add_brightness)
{
	const int mask_index = rand() % num_brightness_shadow_mask_images_;
	const std::string mask_filename = brightness_shadow_mask_filenames_[mask_index];
	cv::Mat mask = cv::imread(mask_filename, cv::IMREAD_GRAYSCALE);

	cv::resize(mask, mask, cv::Size(blended_image.cols, blended_image.rows));

	// rotate, scale, and shift the mask randomly
	const double rotation_angle = rand() % 360;		// in [deg] !
	const double scale_factor = (rand() % 71 + 80) * 0.01;		// resize ratio in range 0.8 to 1.5
	const int interpolation_mode = (scale_factor < 1. ? cv::INTER_AREA : (scale_factor > 1. ? cv::INTER_CUBIC : cv::INTER_LINEAR));	// scale_factor=1.0: CV_INTER_LINEAR,  scale_factor<1.0: CV_INTER_AREA,  scale_factor>1.0: CV_INTER_CUBIC
	const double translation_factor_x = (rand() % 100 - 50) * 0.01;		// in [-0.5, 0.5]
	const double translation_factor_y = (rand() % 100 - 50) * 0.01;		// in [-0.5, 0.5]
	const cv::Point translation_offset(translation_factor_x*mask.cols, translation_factor_y*mask.rows);
	rotateIlluminationMask(mask, rotation_angle, scale_factor, translation_offset, interpolation_mode);

	// add mask to image
	const float f = (15 + rand() % 18) *0.01;		//  opacity factor as in Gimp, in [0.15, 0.33]		// todo: param
	const int blur_kernel_size = 2*static_cast<int>((double)(31 + rand()%(71-31)) * 0.5 * (double)blended_image.cols/1280.) + 1;	// use blur in range of kernel size [31, 71] for image width 1280
	const double blur_sigma = 0.3*((blur_kernel_size-1.)*0.5 - 1.) + 0.8;
	if (add_brightness == false)		// false for shadow
	{
		// add as shadow
		mask.convertTo(mask, blended_image.type());
		cv::GaussianBlur(mask, mask, cv::Size(blur_kernel_size, blur_kernel_size), blur_sigma, blur_sigma, cv::BORDER_CONSTANT);

		// blended_image = blended_image - f*mask;
		std::vector<cv::Mat> rgb;
		cv::split(blended_image, rgb);
		rgb[0] = rgb[0] - f * mask;
		rgb[1] = rgb[1] - f * mask;
		rgb[2] = rgb[2] - f * mask;
		cv::merge(rgb, blended_image);
	}
	else
	{
		// add as brightness
		mask.convertTo(mask, blended_image.type());
		cv::GaussianBlur(mask, mask, cv::Size(blur_kernel_size, blur_kernel_size), blur_sigma, blur_sigma, cv::BORDER_CONSTANT);

		// blended_image = (1-f)*blended_image + f*mask;
		std::vector<cv::Mat> rgb;
		cv::split(blended_image, rgb);
		rgb[0] = (1-f) * rgb[0] + f * mask;
		rgb[1] = (1-f) * rgb[1] + f * mask;
		rgb[2] = (1-f) * rgb[2] + f * mask;
		cv::merge(rgb, blended_image);
	}
}


void ipa_dirt_detection_dataset_tools::ImageBlender::addIlluminationFromTemplate(cv::Mat& blended_image)
{
	const int mask_index = rand() % num_illumination_mask_images_;
	const std::string mask_filename = illumination_mask_filenames_[mask_index];
	cv::Mat mask = cv::imread(mask_filename, cv::IMREAD_GRAYSCALE);

	cv::resize(mask, mask, cv::Size(blended_image.cols, blended_image.rows));

	// rotate, scale, and shift the mask randomly
	const double rotation_angle = rand() % 360;		// in [deg] !
	const double scale_factor = (rand() % 71 + 80) * 0.01;		// resize ratio in range 0.8 to 1.5
	const int interpolation_mode = (scale_factor < 1. ? cv::INTER_AREA : (scale_factor > 1. ? cv::INTER_CUBIC : cv::INTER_LINEAR));	// scale_factor=1.0: CV_INTER_LINEAR,  scale_factor<1.0: CV_INTER_AREA,  scale_factor>1.0: CV_INTER_CUBIC
	const double translation_factor_x = (rand() % 100 - 50) * 0.01;		// in [-0.5, 0.5]
	const double translation_factor_y = (rand() % 100 - 50) * 0.01;		// in [-0.5, 0.5]
	const cv::Point translation_offset(translation_factor_x*mask.cols, translation_factor_y*mask.rows);
	rotateIlluminationMask(mask, rotation_angle, scale_factor, translation_offset, interpolation_mode);

	// add illumination
	const float f = (15 + rand() % 18) *0.01;		//  opacity factor as in Gimp, in [0.15, 0.33]		// todo: param
	mask.convertTo(mask, blended_image.type());
	const int blur_kernel_size = 2*static_cast<int>((double)(75 + rand()%(501-75)) * 0.5 * (double)blended_image.cols/1280.) + 1;	// use blur in range of kernel size [75, 501] for image width 1280
	const double blur_sigma = 0.3*((blur_kernel_size-1.)*0.5 - 1.) + 0.8;
	cv::GaussianBlur(mask, mask, cv::Size(blur_kernel_size, blur_kernel_size), blur_sigma, blur_sigma, cv::BORDER_CONSTANT);

	// blended_image = (1-f)*blended_image + f*mask;
	std::vector<cv::Mat> rgb;
	cv::split(blended_image, rgb);
	rgb[0] = (1-f) * rgb[0] + f * mask;
	rgb[1] = (1-f) * rgb[1] + f * mask;
	rgb[2] = (1-f) * rgb[2] + f * mask;
	cv::merge(rgb, blended_image);
}


void ipa_dirt_detection_dataset_tools::ImageBlender::resizeDirt(cv::Mat& dirt_image, cv::Mat& dirt_mask)
{
//	std::cout << dirt_image.cols << ' ' << dirt_image.rows << std::endl;
//	std::cout << dirt_mask.cols << ' ' << dirt_mask.rows << std::endl;
	double resize_ratio = (rand() % 4 + 8) / 10.0;		// resize ratio in range 0.8 to 1.2
	//std::cout << "resize ratio: " << resize_ratio << std::endl;
	int interpolation_mode = cv::INTER_AREA;		// best for shrinking images
	if (resize_ratio > 1.)
		interpolation_mode = cv::INTER_CUBIC;	// best for enlarging images
	cv::resize(dirt_image, dirt_image, cv::Size(0, 0), resize_ratio, resize_ratio, interpolation_mode);
	cv::resize(dirt_mask, dirt_mask, cv::Size(0, 0), resize_ratio, resize_ratio, interpolation_mode);
}
