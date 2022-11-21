#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include "estimator.h"
#include "parameters.h"
#include "utility/visualization.h"


Estimator estimator;

std::condition_variable con;//锁专用
double current_time = -1;
queue<sensor_msgs::ImuConstPtr> imu_buf;//存放IMU信息
queue<sensor_msgs::PointCloudConstPtr> feature_buf;

// global variable saving the lidar odometry
deque<nav_msgs::Odometry> odomQueue;
odometryRegister *odomRegister;

std::mutex m_buf;
std::mutex m_state;
std::mutex m_estimator;
std::mutex m_odom;

double latest_time;
Eigen::Vector3d tmp_P;
Eigen::Quaterniond tmp_Q;
Eigen::Vector3d tmp_V;
Eigen::Vector3d tmp_Ba;
Eigen::Vector3d tmp_Bg;
Eigen::Vector3d acc_0;
Eigen::Vector3d gyr_0;
bool init_feature = 0;
bool init_imu = 1;
double last_imu_t = 0;

//根据imu的数据，通过中值定理进行新的位姿预测，并发布最新里程计
//根据上一帧的PQV，积分得到当前帧的PQV
void predict(const sensor_msgs::ImuConstPtr &imu_msg)
{
    double t = imu_msg->header.stamp.toSec();
    //判断第一帧
    if (init_imu)
    {
        latest_time = t;
        init_imu = 0;
        return;
    }
    //计算当前imu距离上一时刻的时间间隔
    double dt = t - latest_time;
    latest_time = t;
    //加速度信息
    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    Eigen::Vector3d linear_acceleration{dx, dy, dz};
    //角速度
    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Eigen::Vector3d angular_velocity{rx, ry, rz};

    //中值定理计算pvq
    //
    Eigen::Vector3d un_acc_0 = tmp_Q * (acc_0 - tmp_Ba) - estimator.g;
   
    Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - tmp_Bg;
    //求Q
    tmp_Q = tmp_Q * Utility::deltaQ(un_gyr * dt);
    //加速度
    Eigen::Vector3d un_acc_1 = tmp_Q * (linear_acceleration - tmp_Ba) - estimator.g;

    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
    //V速度和P位姿
    tmp_P = tmp_P + dt * tmp_V + 0.5 * dt * dt * un_acc;
    tmp_V = tmp_V + dt * un_acc;

    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

void update()
{
    TicToc t_predict;
    latest_time = current_time;
    tmp_P = estimator.Ps[WINDOW_SIZE];
    tmp_Q = estimator.Rs[WINDOW_SIZE];
    tmp_V = estimator.Vs[WINDOW_SIZE];
    tmp_Ba = estimator.Bas[WINDOW_SIZE];
    tmp_Bg = estimator.Bgs[WINDOW_SIZE];
    acc_0 = estimator.acc_0;
    gyr_0 = estimator.gyr_0;

    queue<sensor_msgs::ImuConstPtr> tmp_imu_buf = imu_buf;
    for (sensor_msgs::ImuConstPtr tmp_imu_msg; !tmp_imu_buf.empty(); tmp_imu_buf.pop())
        predict(tmp_imu_buf.front());
}

//这个函数的作用就是 把图像帧 和 对应的IMU数据们 配对起来,而且IMU数据时间是在图像帧的前面
std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>>
getMeasurements()
{
    std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;

    while (ros::ok())//边界判断：数据取完了，说明配对完成
    {
        //两个中有一个为空则跳出循环
        if (imu_buf.empty() || feature_buf.empty())
            return measurements;
        
        //边界判断：IMU buf里面所有数据的时间戳都比img buf第一个帧时间戳要早，说明缺乏IMU数据，需要等待IMU数据
        //imu 太慢 ，整个imu太老，等新的imu数据
        if (!(imu_buf.back()->header.stamp.toSec() > feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            return measurements;//统计等待的次数
        }
        //边界判断：IMU第一个数据的时间要大于第一个图像特征数据的时间(说明图像帧有多的)
        //图像太老，扔掉最老图(有取反)
        if (!(imu_buf.front()->header.stamp.toSec() < feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            ROS_WARN("throw img, only should happen at the beginning");
            feature_buf.pop();
            continue; 
        }
        //核心操作：装入视觉帧信息
        sensor_msgs::PointCloudConstPtr img_msg = feature_buf.front();
        feature_buf.pop();
        //核心操作：转入IMU信息
        std::vector<sensor_msgs::ImuConstPtr> IMUs;
        while (imu_buf.front()->header.stamp.toSec() < img_msg->header.stamp.toSec() + estimator.td)
        {
            IMUs.emplace_back(imu_buf.front());
            imu_buf.pop();
        }
        //注意：把最后一个IMU帧又放回到imu_buf里去了
        //原因：最后一帧IMU信息是被相邻2个视觉帧共享的
        IMUs.emplace_back(imu_buf.front());
        if (IMUs.empty())
            ROS_WARN("no imu between two image");
        measurements.emplace_back(IMUs, img_msg);
    }
    return measurements;
}

//将imu_msg保存到imu_buf，
//IMU状态递推并发布[P,Q,V,header, failureCount]
//就是如果当前处于非线性优化阶段的话，需要把第二件事计算得到的PVQ发布到rviz里去

//imu_callback：获取IMU原始信息，中值积分算位姿，用于下一帧图像未到来之前的高频位姿；
void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    // 检查时间：判断这一帧的时间是否小于等于上一帧的时间，如果结果为true，则说明乱序，时间顺序错误
    if (imu_msg->header.stamp.toSec() <= last_imu_t)
    {
        ROS_WARN("imu message in disorder!");
        return;
    }
    last_imu_t = imu_msg->header.stamp.toSec();

    //通过加锁，防止多线程访问IMU数据缓存队列出现问题。并在取出数据之后，通知主线程process，从阻塞状态唤醒。唤醒的具体功能
    m_buf.lock();
    imu_buf.push(imu_msg);
    m_buf.unlock();
    con.notify_one();

    last_imu_t = imu_msg->header.stamp.toSec();

    {//先将imu数据进行递推用于可视化发布。不准的
        std::lock_guard<std::mutex> lg(m_state);
        // 此处每收到一个imu数据，则通过中值积分对最新的位姿进行预测，并发布最新的里程计(和IMU同频率)；
        predict(imu_msg);
        std_msgs::Header header = imu_msg->header;
        // 发布最新的由IMU直接递推得到的PQV
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            pubLatestOdometry(tmp_P, tmp_Q, tmp_V, header, estimator.failureCount);//p imu 位置  Q  四元数姿态 ，V速度
    }
}

//odom_callback：获取雷达里程计odom信息，转相机系用于VIO初始化，简单的把重定位后的里程计信息放入缓存队列中。
void odom_callback(const nav_msgs::Odometry::ConstPtr& odom_msg)
{
    m_odom.lock();
    odomQueue.push_back(*odom_msg);
    m_odom.unlock();
}

//把cur帧的所有特征点放到feature_buf里，同样需要上锁。注意，cur帧的所有特征点都是整合在一个数据里的，
//feature_callback：获取feature_tracker_node输出的3D特征点；对于第一帧图像是认为没有光流速度的，所以，没有办法追踪并找到特征点。后面的话，也是把消息放到缓冲区内
void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg)
{
    if (!init_feature)//跳过第一帧
    {
        init_feature = 1;
        return;
    }
    m_buf.lock();
    feature_buf.push(feature_msg);//放入buffer队列中,需要上锁
    m_buf.unlock();
    con.notify_one();
}

//restart_callback：如果相机流不稳定，重启VIO部分；
//把所有状态量归零，把buf里的数据全部清空。
void restart_callback(const std_msgs::BoolConstPtr &restart_msg)
{
    if (restart_msg->data == true)
    {
        ROS_WARN("restart the estimator!");
        m_buf.lock();
        while(!feature_buf.empty())
            feature_buf.pop();
        while(!imu_buf.empty())
            imu_buf.pop();
        m_buf.unlock();
        m_estimator.lock();
        estimator.clearState();
        estimator.setParameter();
        m_estimator.unlock();
        current_time = -1;
        last_imu_t = 0;
    }
    return;
}

// 视觉惯性里程计(核心)
//process()：处理IMU，odom，feature信息，完成VIO融合估计位姿。
void process()
{
    while (ros::ok())//process里面只有一个while循环，运行条件为ros::ok()
    {
        //定义一个数据结构
        /*
        数据结构: measurements
        1、首先，measurements他自己就是一个vector；
        2、对于measurements中的每一个measurement，又由2部分组成；
        3、第一部分，由sensor_msgs::ImuConstPtr组成的vector；
        4、第二部分，一个sensor_msgs::PointCloudConstPtr；
        5、这两个sensor_msgs见3.1-6部分介绍。
        6、为什么要这样配对(一个PointCloudConstPtr配上若干个ImuConstPtr)？
        因为IMU的频率比视觉帧的发布频率要高，所以说在这里，需要把一个视觉帧和之前的一串IMU帧的数据配对起来。 
        */
        //    存储imu向量和当前图像帧   的组合数据格式measurements;
        std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;
       
        //多线程上锁不冲突
        std::unique_lock<std::mutex> lk(m_buf);
        con.wait(lk, [&]
                 {//对齐时间
            return (measurements = getMeasurements()).size() != 0;//当前时刻特征和imu数据
                 });
        lk.unlock();

        m_estimator.lock();
        //循环processIMU->push_back->propagate->midPointIntegration
        //2、对measurements中的每一个measurement (IMUs,IMG)组合进行操作
        for (auto &measurement : measurements)
        {
            //2.1、对于measurement中的每一个imu_msg，计算dt并执行processIMU()
            auto img_msg = measurement.second;//取出图像

            // 1. IMU pre-integration imu预积分
            double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
            for (auto &imu_msg : measurement.first)
            {   
                //IMU时间戳
                double t = imu_msg->header.stamp.toSec();
                //相机和IMU同步校准得到的时间差
                double img_t = img_msg->header.stamp.toSec() + estimator.td;
                //对于大多数情况，IMU的时间戳都会比img的早，此时直接选取IMU的数据就行
                if (t <= img_t)
                { 
                    //对于比图像早的imu
                    if (current_time < 0)//第一帧时间
                        current_time = t;
                    double dt = t - current_time;
                    ROS_ASSERT(dt >= 0);
                    current_time = t;
                    dx = imu_msg->linear_acceleration.x;
                    dy = imu_msg->linear_acceleration.y;
                    dz = imu_msg->linear_acceleration.z;
                    rx = imu_msg->angular_velocity.x;
                    ry = imu_msg->angular_velocity.y;
                    rz = imu_msg->angular_velocity.z;
                    //这里干了2件事，IMU粗略地预积分，然后把值传给一个新建的IntegrationBase对象
                    // 计算当前图像帧间的imu预积分值，即帧间平移，旋转，速度，以及bias；
                    // 并利用imu对系统最新状态进行传播，为视觉三角化及重投影提供位姿初值；
                    estimator.processIMU(dt, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                    //对当前图像进行预测，同时得到上一时刻和当前的运动增量
                    //printf("imu: dt:%f a: %f %f %f w: %f %f %f\n",dt, dx, dy, dz, rx, ry, rz);
                }
                 //对于处于边界位置的IMU数据，是被相邻两帧共享的，而且对前一帧的影响会大一些，在这里，对数据线性分配
                else//t>img_t 差值处理//每个大于图像帧时间戳的第一个imu_msg是被两个图像帧共用的(出现次数少)
                {   
                    //两个图像帧公用一个imu数据
                    double dt_1 = img_t - current_time;
                    double dt_2 = t - img_t;
                    current_time = img_t;
                    ROS_ASSERT(dt_1 >= 0);
                    ROS_ASSERT(dt_2 >= 0);
                    ROS_ASSERT(dt_1 + dt_2 > 0);
                    double w1 = dt_2 / (dt_1 + dt_2);
                    double w2 = dt_1 / (dt_1 + dt_2);
                    dx = w1 * dx + w2 * imu_msg->linear_acceleration.x;
                    dy = w1 * dy + w2 * imu_msg->linear_acceleration.y;
                    dz = w1 * dz + w2 * imu_msg->linear_acceleration.z;
                    rx = w1 * rx + w2 * imu_msg->angular_velocity.x;
                    ry = w1 * ry + w2 * imu_msg->angular_velocity.y;
                    rz = w1 * rz + w2 * imu_msg->angular_velocity.z;
                    estimator.processIMU(dt_1, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                    //printf("dimu: dt:%f a: %f %f %f w: %f %f %f\n",dt_1, dx, dy, dz, rx, ry, rz);
                }
            }

            // 2. VINS Optimization
            // TicToc t_s;
            // 每帧图像处理后的特征信息，由一个map的数据结构组成；
            // 键为特征ID，值为该特征在每个相机中的x,y,z,u,v,velocity_x,velocity_y,depth 8个变量;
            // map< 特征ID, vector< pair<相机ID, x,y,z,u,v,vel_x,vel_y,depth >>>
            map<int, vector<pair<int, Eigen::Matrix<double, 8, 1>>>> image;
            for (unsigned int i = 0; i < img_msg->points.size(); i++)
            {   //通道
                int v = img_msg->channels[0].values[i] + 0.5;
                int feature_id = v / NUM_OF_CAM;
                int camera_id = v % NUM_OF_CAM;
                double x = img_msg->points[i].x;
                double y = img_msg->points[i].y;
                double z = img_msg->points[i].z;
                double p_u = img_msg->channels[1].values[i];
                double p_v = img_msg->channels[2].values[i];
                double velocity_x = img_msg->channels[3].values[i];
                double velocity_y = img_msg->channels[4].values[i];
                double depth = img_msg->channels[5].values[i];
 
                ROS_ASSERT(z == 1);
                Eigen::Matrix<double, 8, 1> xyz_uv_velocity_depth;
                xyz_uv_velocity_depth << x, y, z, p_u, p_v, velocity_x, velocity_y, depth;
                image[feature_id].emplace_back(camera_id,  xyz_uv_velocity_depth);
            }

            // Get initialization info from lidar odometry
            vector<float> initialization_info;
            m_odom.lock();
            initialization_info = odomRegister->getOdometry(odomQueue, img_msg->header.stamp.toSec() + estimator.td);
            m_odom.unlock();

            // initialization_info为从激光里程计接收到的IMU频率的里程计；
            estimator.processImage(image, initialization_info, img_msg->header);
            // double whole_t = t_s.toc();
            // printStatistics(estimator, whole_t);

            // 3. Visualization
            std_msgs::Header header = img_msg->header;
            pubOdometry(estimator, header);       //pub 当前最新滑窗VIO位姿，并且写入到文件；
            pubKeyPoses(estimator, header);       //pub 滑动窗口内关键帧（10）位姿；
            pubCameraPose(estimator, header);     //pub 相机相对于世界坐标系的位姿（与vio位姿只差个相机与imu之间的外参）；
            pubPointCloud(estimator, header);     //pub 滑动窗口中世界坐标系下的点云信息；
            pubTF(estimator, header);             //pub tf
            pubKeyframe(estimator);               //pub 最新加入的关键帧位姿及其点云信息；（初始化及机器人静止时没有关键帧加入，则不会pub）
        }
        m_estimator.unlock();

        m_buf.lock();
        m_state.lock();
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            update();
        m_state.unlock();
        m_buf.unlock();
    }
}


int main(int argc, char **argv)//从上一部分读取特征点
{
    ros::init(argc, argv, "vins");
    ros::NodeHandle n;
    ROS_INFO("\033[1;32m----> Visual Odometry Estimator Started.\033[0m");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Warn);

    readParameters(n);//读取参数，由于发布的内容较多，因此，这个节点的发布工作由readParameters(n)函数
    estimator.setParameter();// 设置参数,这里其中一个是设置计时器的参数。另外是，设置视觉测量残差的协方差矩阵

    registerPub(n);// 注册发布的话题,发布用于RVIZ显示的Topic

    odomRegister = new odometryRegister(n);//把里程计信息从lidar帧的坐标系转到VINS视觉图像帧的坐标系

    ros::Subscriber sub_imu     = n.subscribe(IMU_TOPIC,      5000, imu_callback,  ros::TransportHints().tcpNoDelay());//imu 消息
    ros::Subscriber sub_odom    = n.subscribe("odometry/imu", 5000, odom_callback);//激光雷达发出的
    ros::Subscriber sub_image   = n.subscribe(PROJECT_NAME + "/vins/feature/feature", 1, feature_callback);
    ros::Subscriber sub_restart = n.subscribe(PROJECT_NAME + "/vins/feature/restart", 1, restart_callback);//重启
    if (!USE_LIDAR)
        sub_odom.shutdown();
         
    // 创建视觉估计模块的     *******主线程process    
    //使用C++线程库来开始一个线程需要构造一个std::thread对象，
    //std::thread可以与任何可调用类型一同工作。一旦开启了线程，需要显式的决定是等待其结束（join）还是让他在后台运行（detach）。
    std::thread measurement_process{process};
    // 分配剩余线程
    ros::MultiThreadedSpinner spinner(4);
    spinner.spin();

    return 0;
}