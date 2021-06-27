#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <vector>
#include <algorithm>

//This class takes a LaserScan as an input, and filter artefact points that are too close
class FilterLidar {
    protected:
        ros::Subscriber pc_sub_;
        ros::Publisher filtered_pc_pub_;

        ros::NodeHandle nh_;
        std::string base_frame_;
        float min_range_;
    protected:

        void pc_callback(const sensor_msgs::LaserScanPtr msg) {
            std::vector<float> ranges = msg->ranges;
            std::vector<float> intensities = msg->intensities;
            //std::for_each(ranges.begin(), ranges.end(),[&idx, this](float& val){
            //    ++idx;
            //    if (val <= this->min_range_ ) val=0.0;});
            //intensities.at(idx)=0;
            for (size_t idx =0; idx < ranges.size(); idx++) {
                if (ranges[idx] <= min_range_ ) {
                    ranges[idx]=0.0;
                    intensities[idx]=0.0;
                }
            }

            sensor_msgs::LaserScan output;
            output.header.frame_id = base_frame_;
            output.header.stamp = ros::Time::now();
            output.angle_min = msg->angle_min;
            output.angle_max = msg->angle_max;
            output.angle_increment = msg->angle_increment;
            output.time_increment = msg->time_increment;
            output.scan_time = msg->scan_time;
            output.range_min = min_range_;
            output.range_max = msg->range_max;
            output.ranges = ranges;
            output.intensities = intensities;
            
            filtered_pc_pub_.publish(output);

        }

    public:
        FilterLidar() : nh_("~") {

            nh_.param("base_frame",base_frame_,std::string("base_link"));
            nh_.param("min_range",min_range_,float(.2));

            pc_sub_ = nh_.subscribe("input_scan",1,&FilterLidar::pc_callback,this);
            filtered_pc_pub_ = nh_.advertise<sensor_msgs::LaserScan>("output_scan",1);
        }

};

int main(int argc, char * argv[]) 
{
    ros::init(argc,argv,"filter_lidar");
    FilterLidar of;
    ros::spin();
}

