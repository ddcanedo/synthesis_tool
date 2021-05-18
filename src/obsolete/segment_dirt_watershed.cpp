#include "ipa_dirt_detection_dataset_tools/obsolete/segment_dirt_watershed.h"


ipa_dirt_detection_dataset_tools::SegmentDirtWatershed::SegmentDirtWatershed(const std::string dirt_image_path, const std::string cropped_image_path, const std::string cropped_mask_path, const bool background_color,
		const int crop_residual)
{
	dirt_image_path_ = dirt_image_path;
	cropped_image_path_ = cropped_image_path;
	cropped_mask_path_ = cropped_mask_path;

	background_color_ = background_color;
	crop_residual_ = crop_residual;

	std::cout << "There three paths are: " << dirt_image_path_ << std::endl << cropped_image_path_ << std::endl << cropped_mask_path_ << std::endl;
}

ipa_dirt_detection_dataset_tools::SegmentDirtWatershed::~SegmentDirtWatershed()
{
}

void ipa_dirt_detection_dataset_tools::SegmentDirtWatershed::run()
{
	boost::filesystem::path dirt_images(dirt_image_path_);
	boost::filesystem::directory_iterator end_itr;

	for (boost::filesystem::directory_iterator itr(dirt_images); itr != end_itr; ++itr)
	{
		const boost::filesystem::directory_entry entry = *itr;
		std::string path = entry.path().string();
		std::cout << "Current Path to dirt frame is: " << path << std::endl;

		dirt_frame_ = cv::imread(path, CV_LOAD_IMAGE_COLOR);
		segment();
		crop();

		examinate();    // optional, for result visulization

		std::vector<std::string> strs;              // split the image name from the data path
		std::string file_name;
		boost::split(strs, path, boost::is_any_of("\t,/"));
		for (std::vector<std::string>::iterator it = strs.begin(); it != strs.end(); it++)
		{
			std::cout << *it << std::endl;
			file_name = *(it);
		}
		strs.clear();
		boost::split(strs, file_name, boost::is_any_of("\t,."));
		std::cout << "image name is: " << strs[0] << std::endl;

		std::string dirt_path = cropped_image_path_ + '/' + strs[0] + "_cropped.jpg";
		std::string mask_path = cropped_mask_path_ + '/' + strs[0] + "_crpooed_mask.bmp";
		std::cout << dirt_path << std::endl;
		std::cout << mask_path << std::endl;
		cv::imwrite(dirt_path, cropped_dirt_frame_);
		cv::imwrite(mask_path, cropped_mask_frame_);
	}
}

void ipa_dirt_detection_dataset_tools::SegmentDirtWatershed::segment()
{
	cv::Mat gray_dirt;
	cv::cvtColor(dirt_frame_, gray_dirt, cv::COLOR_BGR2GRAY);

	if (background_color_ == 1)
		cv::GaussianBlur(gray_dirt, gray_dirt, cv::Size(5, 5), 0, 0);  // remove noises, for white background !!!!

	int cols = gray_dirt.cols;
	int rows = gray_dirt.rows;

	cv::Mat binary_map = cv::Mat::zeros(rows, cols, gray_dirt.type());
	cv::threshold(gray_dirt, binary_map, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	std::cout << binary_map.dims << ' ' << binary_map.cols << binary_map.rows << std::endl;
	if (background_color_ == 0)
		cv::bitwise_not(binary_map, binary_map);             // optional, for black background !!!!!!!!!
	cv::imshow("binary_map", binary_map);
	cv::waitKey(0);

	//cv::Mat kernel = getStructuringElement(cv::MORPH_RECT, Size((2*2) + 1, (2*2)+1));   // dilation the area of the object
	//cv::morphologyEx(binary_map, binary_map, cv::MORPH_CLOSE ,kernel);                  // if useful?

	cv::Mat marker = binary_map.clone();
	marker = marker + 1;
	marker.convertTo(marker, CV_32S);

	cv::watershed(dirt_frame_, marker);

	mask_frame_ = marker.clone();
	for (int i = 0; i < marker.rows; i++)
	{
		for (int j = 0; j < marker.cols; j++)
		{
			int index = marker.at<int>(i, j);
			if (index == -1)
				mask_frame_.at<int>(i, j) = 0;
			else if (index == 1)
				mask_frame_.at<int>(i, j) = 1;
			else
				mask_frame_.at<int>(i, j) = 0;
		}
	}

	//std::cout << mask_frame_;
	mask_frame_.convertTo(mask_frame_, gray_dirt.type());

	//if (background_color_ == 1)
	//{
	//int morph_size = 2;                                             // Optional !!!!!!!!!!!!!!!!! For white backgtround
	//cv::Mat element = cv::getStructuringElement( cv::MORPH_RECT, Size( 2*morph_size + 1, 2*morph_size+1 ));
	//cv::morphologyEx( mask_frame_, mask_frame_, cv::MORPH_CLOSE, element );
	//}
}

void ipa_dirt_detection_dataset_tools::SegmentDirtWatershed::crop()
{
	int cols = mask_frame_.cols;
	int rows = mask_frame_.rows;

	int left_edge = 0;
	int right_edge = 0;
	int upper_edge = 0;
	int down_edge = 0;

	bool left_edge_flag = 0;
	bool upper_edge_flag = 0;

	cv::Mat mask_frame = mask_frame_.clone();       // only convert to 32S, it can be used for check if 0 or 1
	mask_frame.convertTo(mask_frame, CV_32S);

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			// -----------------------------------------------------------------------------
			if (mask_frame.at<int>(i, j) == 1 && left_edge_flag == 0)
			{
				left_edge = j;
				left_edge_flag = 1;
			}
			else if (mask_frame.at<int>(i, j) == 1 && left_edge_flag == 1)
			{
				if (j > left_edge)
					right_edge = j > right_edge ? j : right_edge;
				else
					left_edge = j;
			}
			if (mask_frame.at<int>(i, j) == 1 && upper_edge_flag == 0)
			{
				upper_edge = i;
				upper_edge_flag = 1;
			}
			else if (mask_frame.at<int>(i, j) == 1 && upper_edge_flag == 1)
			{
				if (i > upper_edge)
					down_edge = i > down_edge ? i : down_edge;
				else
					upper_edge = i;
			}
			// ----------------------------------------------------------------------------
		}
	}

	std::cout << left_edge << ' ' << right_edge << std::endl;
	std::cout << upper_edge << ' ' << down_edge << std::endl;

	cv::Mat dirt_frame_pad_array = dirt_frame_.clone();
	cv::Mat mask_frame_pad_array = mask_frame_.clone();

	cv::copyMakeBorder(dirt_frame_, dirt_frame_pad_array, crop_residual_, crop_residual_, crop_residual_, crop_residual_, cv::BORDER_REPLICATE);
	cv::copyMakeBorder(mask_frame_, mask_frame_pad_array, crop_residual_, crop_residual_, crop_residual_, crop_residual_, cv::BORDER_REPLICATE);

	cv::Rect roi;
	roi.x = left_edge;                                          //(left_edge - crop_residual) > 0 ? (left_edge - crop_residual) : 0;
	roi.y = upper_edge;                                         //(upper_edge - crop_residual) > 0 ? (upper_edge - crop_residual) : 0;
	roi.width = abs(right_edge - left_edge) + 2 * crop_residual_;
	roi.height = abs(down_edge - upper_edge) + 2 * crop_residual_;

	cropped_dirt_frame_ = dirt_frame_pad_array(roi);
	cropped_mask_frame_ = mask_frame_pad_array(roi);

	int morph_size = 3;                                             // Optional !!!!!!!!!!!!!!!!!
	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * morph_size + 1, 2 * morph_size + 1));
	cv::morphologyEx(cropped_mask_frame_, cropped_mask_frame_, cv::MORPH_CLOSE, element);

	cv::imshow("cropped_mask_frame", cropped_mask_frame_ * 255);
	cv::waitKey(0);
}

void ipa_dirt_detection_dataset_tools::SegmentDirtWatershed::examinate()
{
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;

	cv::findContours(mask_frame_, contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

	cv::Mat drawing = dirt_frame_.clone();
	for (int i = 0; i < contours.size(); i++)
	{
		cv::Scalar color = cv::Scalar(255, 0, 0);
		cv::drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, cv::Point());
	}

	cv::imshow("drawing", drawing);
	cv::waitKey(0);
}
