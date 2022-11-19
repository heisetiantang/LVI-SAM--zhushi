#pragma once
#ifndef _UTILITY_LIDAR_ODOMETRY_H_
#define _UTILITY_LIDAR_ODOMETRY_H_

#include <ros/ros.h>

#include <std_msgs/Header.h>
#include <std_msgs/Float64MultiArray.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/NavSatFix.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <opencv/cv.h>


#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/search.hpp>
// 一个通用搜索类，所有的搜索包装器都必须继承于它。每种搜索方法必须实现 2 种不同类型的搜索：
// （1）NearKSearch - 搜索 K-最近邻；（2）radiusSearch - 在给定半径的球体中搜索所有最近的邻居。
// 每种搜索方法的输入可以通过 3 种不同的方式给出：
// （1）作为查询点；（2）作为（云，索引）对；（3）作为索引。对于后一个选项，假设用户首先通过setInputCloud () 方法指定了输入。
#include <pcl/range_image/range_image.h>
#include <pcl/kdtree/kdtree_flann.h>//​ KdTreeFLANN是一种使用 kD 树结构的通用 3D 空间定位器。
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/gicp.h>
#include <pcl/io/pcd_io.h>//​ 点云数据（PCD）文件格式阅读器。
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h> 
#include <pcl_conversions/pcl_conversions.h>

#include <tf/LinearMath/Quaternion.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
 
#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>
#include <deque>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cfloat>
#include <iterator>
#include <sstream>
#include <string>
#include <limits>
#include <iomanip>
#include <array>
#include <thread>
#include <mutex>

using namespace std;

typedef pcl::PointXYZI PointType;//关键字 typedef 可以为类型起一个新的别名

//初始化函数，读取参数相关数据，
class ParamServer
{
public:

    ros::NodeHandle nh;
     
    std::string PROJECT_NAME;
    std::string robot_id;
    string pointCloudTopic;
    string imuTopic;
    string odomTopic;
    string gpsTopic;

    // GPS 设置
    bool useImuHeadingInitialization;
    bool useGpsElevation;
    float gpsCovThreshold;
    float poseCovThreshold;

    // Save pcd
    bool savePCD;
    string savePCDDirectory;

    // Velodyne Sensor Configuration: Velodyne// Velodyne 传感器配置：Velodyne
    int N_SCAN;//激光雷达通道的数量(例如,16、32、64、128)
    int Horizon_SCAN;//激光雷达水平分辨率(Velodyne:1800, Ouster:512,1024,2048)
    string timeField;//点时间戳字段. Velodyne - "time", Ouster - "t"
    int downsampleRate;//下载样本频率，default: 1. Downsample your data if too many points. i.e., 16 = 64 / 4, 16 = 16 / 1 

    // IMU
    float imuAccNoise;
    float imuGyrNoise;
    float imuAccBiasN;
    float imuGyrBiasN;
    float imuGravity;
    vector<double> extRotV;
    vector<double> extRPYV;
    vector<double> extTransV;
    Eigen::Matrix3d extRot;//构造3*3 double类型矩阵
    Eigen::Matrix3d extRPY;
    Eigen::Vector3d extTrans;
    Eigen::Quaterniond extQRPY;

    // LOAM
    float edgeThreshold;
    float surfThreshold;
    int edgeFeatureMinValidNum;
    int surfFeatureMinValidNum;

    //********角点和平面点比例值
    //创建一个容器来保存角点和平面点的比例值
    vector<float> ratio_value;
    int rotioofedge2surf;
    float  edgeNum ;
    float  surfaceNum ;
    float  rotiofactoredge;
    float  rotiofactorsurf;
    //*************************




    // voxel filter paprams
    float odometrySurfLeafSize;
    float mappingCornerLeafSize;
    float mappingSurfLeafSize ;

    float z_tollerance; 
    float rotation_tollerance;

    // CPU Params
    int numberOfCores;
    double mappingProcessInterval;

    // Surrounding map
    float surroundingkeyframeAddingDistThreshold; 
    float surroundingkeyframeAddingAngleThreshold; 
    float surroundingKeyframeDensity;
    float surroundingKeyframeSearchRadius;
    
    // Loop closure
    bool loopClosureEnableFlag;
    int   surroundingKeyframeSize;
    float historyKeyframeSearchRadius;
    float historyKeyframeSearchTimeDiff;
    int   historyKeyframeSearchNum;
    float historyKeyframeFitnessScore;

    // global map visualization radius
    float globalMapVisualizationSearchRadius;
    float globalMapVisualizationPoseDensity;
    float globalMapVisualizationLeafSize;
    
    //设置和查询参数，n.paramstd::string>("my_param", s, "default_value");代表将名字为my_param的参数值赋值给s变量，没有参数值时则使用默认参数值
    ParamServer()
    {   
        nh.param<std::string>("/PROJECT_NAME", PROJECT_NAME, "sam");

        nh.param<std::string>("/robot_id", robot_id, "roboat");

        nh.param<std::string>(PROJECT_NAME + "/pointCloudTopic", pointCloudTopic, "points_raw");
        nh.param<std::string>(PROJECT_NAME + "/imuTopic", imuTopic, "imu_correct");
        nh.param<std::string>(PROJECT_NAME + "/odomTopic", odomTopic, "odometry/imu");
        nh.param<std::string>(PROJECT_NAME + "/gpsTopic", gpsTopic, "odometry/gps");

        nh.param<bool>(PROJECT_NAME + "/useImuHeadingInitialization", useImuHeadingInitialization, false);//初始化imu
        nh.param<bool>(PROJECT_NAME + "/useGpsElevation", useGpsElevation, false);//gps优化
        nh.param<float>(PROJECT_NAME + "/gpsCovThreshold", gpsCovThreshold, 2.0);//cov协方差矩阵
        nh.param<float>(PROJECT_NAME + "/poseCovThreshold", poseCovThreshold, 25.0);

        nh.param<bool>(PROJECT_NAME + "/savePCD", savePCD, false);
        nh.param<std::string>(PROJECT_NAME + "/savePCDDirectory", savePCDDirectory, "/tmp/loam/");

        nh.param<int>(PROJECT_NAME + "/N_SCAN", N_SCAN, 16);
        nh.param<int>(PROJECT_NAME + "/Horizon_SCAN", Horizon_SCAN, 1800);
        nh.param<std::string>(PROJECT_NAME + "/timeField", timeField, "time");
        nh.param<int>(PROJECT_NAME + "/downsampleRate", downsampleRate, 1);

        nh.param<float>(PROJECT_NAME + "/imuAccNoise", imuAccNoise, 0.01);//Noise和BiasN噪音参数
        nh.param<float>(PROJECT_NAME + "/imuGyrNoise", imuGyrNoise, 0.001);
        nh.param<float>(PROJECT_NAME + "/imuAccBiasN", imuAccBiasN, 0.0002);
        nh.param<float>(PROJECT_NAME + "/imuGyrBiasN", imuGyrBiasN, 0.00003);
        nh.param<float>(PROJECT_NAME + "/imuGravity", imuGravity, 9.80511);
        // extrinsicTrans: [0.0, 0.0, 0.0]
        // extrinsicRot: [-1, 0, 0, 0, 1, 0, 0, 0, -1]
        // extrinsicRPY: [0, 1, 0, -1, 0, 0, 0, 0, 1]
        nh.param<vector<double>>(PROJECT_NAME+ "/extrinsicRot", extRotV, vector<double>());//外來的rot
        nh.param<vector<double>>(PROJECT_NAME+ "/extrinsicRPY", extRPYV, vector<double>());
        nh.param<vector<double>>(PROJECT_NAME+ "/extrinsicTrans", extTransV, vector<double>());
        //RowMajor，它表明matrix使用按行存储，默认是按列存储。
        //eigen的map功能进行内存映射,转换格式
        //extRot = Eigen::Map(一个模板函数，用于将一串数据映射到一个矩阵或者向量中)
        //<const Eigen::Matrix<double(指定想要生成的矩阵或者向量的类型), -1, -1, Eigen::RowMajor（RowMajor，是按行存储）>>
        //(extRotV.data()(指向数据的指针), 3(行), 3(列));
        extRot = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extRotV.data(), 3, 3);
        extRPY = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extRPYV.data(), 3, 3);
        extTrans = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extTransV.data(), 3, 1);
        extQRPY = Eigen::Quaterniond(extRPY);

        nh.param<float>(PROJECT_NAME + "/edgeThreshold", edgeThreshold, 0.1);//角点阈值
        nh.param<float>(PROJECT_NAME + "/surfThreshold", surfThreshold, 0.1);//表面阈值
        nh.param<int>(PROJECT_NAME + "/edgeFeatureMinValidNum", edgeFeatureMinValidNum, 10);//角点特征最小生效数量
        nh.param<int>(PROJECT_NAME + "/surfFeatureMinValidNum", surfFeatureMinValidNum, 100);//平面点特征最小生效数量

        nh.param<float>(PROJECT_NAME + "/odometrySurfLeafSize", odometrySurfLeafSize, 0.2);
        nh.param<float>(PROJECT_NAME + "/mappingCornerLeafSize", mappingCornerLeafSize, 0.2);
        nh.param<float>(PROJECT_NAME + "/mappingSurfLeafSize", mappingSurfLeafSize, 0.2);

        nh.param<float>(PROJECT_NAME + "/z_tollerance", z_tollerance, FLT_MAX);
        nh.param<float>(PROJECT_NAME + "/rotation_tollerance", rotation_tollerance, FLT_MAX);

        nh.param<int>(PROJECT_NAME + "/numberOfCores", numberOfCores, 2);
        nh.param<double>(PROJECT_NAME + "/mappingProcessInterval", mappingProcessInterval, 0.15);

        nh.param<float>(PROJECT_NAME + "/surroundingkeyframeAddingDistThreshold", surroundingkeyframeAddingDistThreshold, 1.0);//关键帧距离阈值
        nh.param<float>(PROJECT_NAME + "/surroundingkeyframeAddingAngleThreshold", surroundingkeyframeAddingAngleThreshold, 0.2);//关键帧角度阈值
        nh.param<float>(PROJECT_NAME + "/surroundingKeyframeDensity", surroundingKeyframeDensity, 1.0);//关键帧稠密度
        nh.param<float>(PROJECT_NAME + "/surroundingKeyframeSearchRadius", surroundingKeyframeSearchRadius, 50.0);//关键帧搜索半径

        nh.param<bool>(PROJECT_NAME + "/loopClosureEnableFlag", loopClosureEnableFlag, false);//回环标志
        nh.param<int>(PROJECT_NAME + "/surroundingKeyframeSize", surroundingKeyframeSize, 50);//关键帧尺寸
        nh.param<float>(PROJECT_NAME + "/historyKeyframeSearchRadius", historyKeyframeSearchRadius, 10.0);//历史关键帧搜素半径
        nh.param<float>(PROJECT_NAME + "/historyKeyframeSearchTimeDiff", historyKeyframeSearchTimeDiff, 30.0);//历史关键帧搜索时间间隔
        nh.param<int>(PROJECT_NAME + "/historyKeyframeSearchNum", historyKeyframeSearchNum, 25);//历史关键帧搜索数量
        nh.param<float>(PROJECT_NAME + "/historyKeyframeFitnessScore", historyKeyframeFitnessScore, 0.3);//历史关键帧适合度得分

        nh.param<float>(PROJECT_NAME + "/globalMapVisualizationSearchRadius", globalMapVisualizationSearchRadius, 1e3);
        nh.param<float>(PROJECT_NAME + "/globalMapVisualizationPoseDensity", globalMapVisualizationPoseDensity, 10.0);
        nh.param<float>(PROJECT_NAME + "/globalMapVisualizationLeafSize", globalMapVisualizationLeafSize, 1.0);

        usleep(100);
    }
    
    //类函数imuConverter：从原始的imu信息中，计算旋转加速度、旋转陀螺仪、转动、滚动、俯仰、偏转等信息
    sensor_msgs::Imu imuConverter(const sensor_msgs::Imu& imu_in)
    {
        sensor_msgs::Imu imu_out = imu_in;//设置输出初始值等于输入值
        // rotate acceleration 加速度计转换
        Eigen::Vector3d acc(imu_in.linear_acceleration.x, imu_in.linear_acceleration.y, imu_in.linear_acceleration.z);
        acc = extRot * acc;
        imu_out.linear_acceleration.x = acc.x();
        imu_out.linear_acceleration.y = acc.y();
        imu_out.linear_acceleration.z = acc.z();
        // rotate gyroscope 陀螺仪转换
        Eigen::Vector3d gyr(imu_in.angular_velocity.x, imu_in.angular_velocity.y, imu_in.angular_velocity.z);
        gyr = extRot * gyr;
        imu_out.angular_velocity.x = gyr.x();
        imu_out.angular_velocity.y = gyr.y();
        imu_out.angular_velocity.z = gyr.z();
        // rotate roll pitch yaw 翻滾、俯仰、偏航转换
        Eigen::Quaterniond q_from(imu_in.orientation.w, imu_in.orientation.x, imu_in.orientation.y, imu_in.orientation.z);
        Eigen::Quaterniond q_final = q_from * extQRPY;
        imu_out.orientation.x = q_final.x();
        imu_out.orientation.y = q_final.y();
        imu_out.orientation.z = q_final.z();
        imu_out.orientation.w = q_final.w();
        //判断是不是九自由度的传感器
        if (sqrt(q_final.x()*q_final.x() + q_final.y()*q_final.y() + q_final.z()*q_final.z() + q_final.w()*q_final.w()) < 0.1)
        {
            ROS_ERROR("Invalid quaternion, please use a 9-axis IMU!");
            ros::shutdown();
        }

        return imu_out;
    }
};

//发布云点图的模板函数
template<typename T>
sensor_msgs::PointCloud2 publishCloud(ros::Publisher *thisPub, T thisCloud, ros::Time thisStamp, std::string thisFrame)
{   
    sensor_msgs::PointCloud2 tempCloud;//将此时的点云数据给新命名的tempCloud，
    pcl::toROSMsg(*thisCloud, tempCloud);//将点云数据转换成ROS的点云数据
    tempCloud.header.stamp = thisStamp;//传入时间戳
    tempCloud.header.frame_id = thisFrame;//传入id
    if (thisPub->getNumSubscribers() != 0)//根据是否有节点订阅所需话题，决定是否发布对应话题,getNumSubscribers判断订阅者是否连接
        thisPub->publish(tempCloud);
    return tempCloud;
}

//获取ros中的时间信息
template<typename T>
double ROS_TIME(T msg)
{
    return msg->header.stamp.toSec();
}

//把从imu传感器中的数据转变成ros中的数据
template<typename T>//角度
void imuAngular2rosAngular(sensor_msgs::Imu *thisImuMsg, T *angular_x, T *angular_y, T *angular_z)
{
    *angular_x = thisImuMsg->angular_velocity.x;
    *angular_y = thisImuMsg->angular_velocity.y;
    *angular_z = thisImuMsg->angular_velocity.z;
}

//将IMU的加速度转换为ros格式的加速度
template<typename T>
void imuAccel2rosAccel(sensor_msgs::Imu *thisImuMsg, T *acc_x, T *acc_y, T *acc_z)
{
    *acc_x = thisImuMsg->linear_acceleration.x;
    *acc_y = thisImuMsg->linear_acceleration.y;
    *acc_z = thisImuMsg->linear_acceleration.z;
}

//将IMU的翻滾、俯仰、偏航转换为ros格式的翻滾、俯仰、偏航
template<typename T>
void imuRPY2rosRPY(sensor_msgs::Imu *thisImuMsg, T *rosRoll, T *rosPitch, T *rosYaw)
{
    double imuRoll, imuPitch, imuYaw;
    tf::Quaternion orientation;
    tf::quaternionMsgToTF(thisImuMsg->orientation, orientation);//把geomsg形式的四元数转化为tf形式，得orientation
    tf::Matrix3x3(orientation).getRPY(imuRoll, imuPitch, imuYaw);

    *rosRoll = imuRoll;
    *rosPitch = imuPitch;
    *rosYaw = imuYaw;
}

// 以及到零点的距离
float pointDistance(PointType p)
{
    return sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
}

// 计算在点云图中的两点之间的距离
float pointDistance(PointType p1, PointType p2)
{
    return sqrt((p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y) + (p1.z-p2.z)*(p1.z-p2.z));
}

#endif