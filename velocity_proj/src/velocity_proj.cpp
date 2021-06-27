#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/TwistStamped.h>
#include <math.h>

class VelProjector{
  private:
    ros::NodeHandle nh_;
    ros::Publisher velocity_pub_;
    ros::Subscriber odom_sub_;
    void odomCallback(const nav_msgs::OdometryConstPtr& data);
    geometry_msgs::TwistStamped twist; 

  public: 
    VelProjector();
};

VelProjector::VelProjector() : nh_("~"){
  odom_sub_ = nh_.subscribe("odom",1,&VelProjector::odomCallback,this);
  velocity_pub_ = nh_.advertise<geometry_msgs::TwistStamped>("robot_frame",1,true);  
}

void VelProjector::odomCallback(const nav_msgs::OdometryConstPtr& data) {
  float siny_cosp = 2 * (data->pose.pose.orientation.w * data->pose.pose.orientation.z + data->pose.pose.orientation.x * data->pose.pose.orientation.y);
  float cosy_cosp = 1 - 2 * (data->pose.pose.orientation.y * data->pose.pose.orientation.y + data->pose.pose.orientation.z * data->pose.pose.orientation.z);
  float yaw = std::atan2(siny_cosp, cosy_cosp);
  twist.header = data->header;
  twist.twist.linear.x = std::cos(yaw)*data->twist.twist.linear.x + std::sin(yaw)*data->twist.twist.linear.y;
  twist.twist.linear.y = std::cos(yaw+M_PI/2)*data->twist.twist.linear.x + std::sin(yaw+M_PI/2)*data->twist.twist.linear.y;
  twist.twist.angular.z = data->twist.twist.angular.z;
  velocity_pub_.publish(twist);
}

int main(int argc, char * argv[]) {
    ros::init(argc,argv,"velocity_projector");
    VelProjector VP;
    ros::spin();
}

