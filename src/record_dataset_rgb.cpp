#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

#include <ipa_dirt_detection_dataset_tools/record_dataset_rgb.h>

using namespace ipa_DatasetCreate;

DatasetCreate::DatasetCreate(ros::NodeHandle node_handle) :
		node_handle_(node_handle)
{
	ros::NodeHandle pnh("~");
	pnh.param("base_path", base_path_, std::string("/home/robot/floors/"));
	std::cout << "base_path_=" << base_path_ << "\n" << std::endl;

	// set current floor index
	current_floor_index_ = 0;
	selectNextFreeFloorIndex();
	current_image_index_ = 0;
	selectNextFreeImageIndex();
	std::cout << "c = capture image\nn = new floor type\nf = select floor type number\nq = quit\n" << std::endl;
	std::cout << "Current floor type " << current_floor_index_ << " , image index " << current_image_index_ << std::endl;
}

DatasetCreate::~DatasetCreate()
{
}

void DatasetCreate::init()
{
	frame_counter_ = 0;
	camera_rgb_image_sub_ = node_handle_.subscribe("/camera/rgb/image_raw", 1, &DatasetCreate::imageCallback, this);
}

void DatasetCreate::imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
	try
	{
		// check if folder exists
		std::stringstream folder;
		folder << base_path_ << "floor_";
		addZerosToFileName(folder, current_floor_index_);
		folder << current_floor_index_ << "/";
		if (boost::filesystem::exists(folder.str()) == false)
			boost::filesystem::create_directory(folder.str());
		current_image_index_ = 0;
		selectNextFreeImageIndex();

		cv::imshow("view", cv_bridge::toCvShare(msg, "bgr8")->image);
		int key = 0;
		key = cv::waitKey(100);

		if (key == 'c')		// capture
		{
			std::stringstream name;
			name << base_path_ << "floor_";
			addZerosToFileName(name, current_floor_index_);
			name << current_floor_index_ << "/img_";
			addZerosToFileName(name, current_image_index_);
			name << current_image_index_ << ".png";
			std::cout << "saving image " << name.str() << std::endl;

			cv::imwrite(name.str().c_str(), cv_bridge::toCvShare(msg, "bgr8")->image);
			selectNextFreeImageIndex();
		}
		else if (key == 'n')
		{
			current_floor_index_ = 0;
			selectNextFreeFloorIndex();
			current_image_index_ = 0;
			selectNextFreeImageIndex();
			std::cout << "New floor type " << current_floor_index_ << std::endl;
		}
		else if (key == 'f')
		{
			std::cout << "Select current floor index: ";
			std::cin >> current_floor_index_;
			current_image_index_ = 0;
			selectNextFreeImageIndex();
			std::cout << "Selected floor type " << current_floor_index_ << " , image index " << current_image_index_ << std::endl;
		}
		else if (key == 'q')
		{
			exit(0);
		}
	}
	catch (cv_bridge::Exception& e)
	{
		ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
	}
}

void DatasetCreate::addZerosToFileName(std::stringstream& name, const int index)
{
	if (index < 10)
		name << "00000";
	else if (index < 100)
		name << "0000";
	else if (index < 1000)
		name << "000";
	else if (index < 10000)
		name << "00";
	else if (index < 100000)
		name << "0";
}

void DatasetCreate::selectNextFreeFloorIndex()
{
	bool current_floor_index_set = false;
	while (current_floor_index_set == false)
	{
		std::stringstream folder;
		folder << base_path_ << "floor_";
		addZerosToFileName(folder, current_floor_index_);
		folder << current_floor_index_ << "/";
		if (boost::filesystem::exists(folder.str()) == false || boost::filesystem::is_empty(folder.str()))
			current_floor_index_set = true;
		else
			current_floor_index_++;
	}
}

void DatasetCreate::selectNextFreeImageIndex()
{
	bool current_image_index_set = false;
	while (current_image_index_set == false)
	{
		std::stringstream file;
		file << base_path_ << "floor_";
		addZerosToFileName(file, current_floor_index_);
		file << current_floor_index_ << "/img_";
		addZerosToFileName(file, current_image_index_);
		file << current_image_index_ << ".png";
		if (boost::filesystem::exists(file.str()) == false)
			current_image_index_set = true;
		else
			current_image_index_++;
	}
}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "record_dataset");
	ros::NodeHandle nh;

	DatasetCreate dc(nh);
	dc.init();

	ros::spin();

}
