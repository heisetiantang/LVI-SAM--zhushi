#include "utility.h"
#include "lvi_sam/cloud_info.h"

// Velodyne,定义了一个结构体点图，根据雷达Velodyne来定义
struct PointXYZIRT
{
    PCL_ADD_POINT4D;// 表示欧几里得 xyz 坐标和强度值的点结构。
    PCL_ADD_INTENSITY;// 激光点反射的强度，也可以存点的索引，里面是一个float 类型的变量
    uint16_t ring;//扫描的激光线
    float time;// 时间
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW// Eigen的字段对齐
} EIGEN_ALIGN16;
// 注册为PCL点云格式，包括的字段为 x,y,z,intensity,ring,time
POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRT,  
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (uint16_t, ring, ring) (float, time, time)
)

// Ouster
// struct PointXYZIRT {
//     PCL_ADD_POINT4D;
//     float intensity;浮动强度
//     uint32_t t;
//     uint16_t reflectivity;反射率
//     uint8_t ring;
//     uint16_t noise;
//     uint32_t range;
//     EIGEN_MAKE_ALIGNED_OPERATOR_NEW
// }EIGEN_ALIGN16;

// POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIRT,
//     (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
//     (uint32_t, t, t) (uint16_t, reflectivity, reflectivity)
//     (uint8_t, ring, ring) (uint16_t, noise, noise) (uint32_t, range, range)
// )

// imu 队列的数据长度
const int queueLength = 500;

class ImageProjection : public ParamServer
{
private:
    // imu队列、odom队列互斥锁
    std::mutex imuLock;
    std::mutex odoLock;

    // 订阅原始激光点云
    ros::Subscriber subLaserCloud;
    ros::Publisher  pubLaserCloud;
    
    // 发布当前帧校正后点云，有效点
    ros::Publisher pubExtractedCloud;
    ros::Publisher pubLaserCloudInfo;

    // imu数据队列（原始数据，转lidar系下）
    ros::Subscriber subImu;
    std::deque<sensor_msgs::Imu> imuQueue;

    // 里程计队列
    ros::Subscriber subOdom;
    std::deque<nav_msgs::Odometry> odomQueue;

    // 激光点云数据队列
    std::deque<sensor_msgs::PointCloud2> cloudQueue;
    // 队列front帧，作为当前处理帧点云
    sensor_msgs::PointCloud2 currentCloudMsg;
    
    // 当前激光帧起止时刻间对应的imu数据，计算相对于起始时刻的旋转增量，以及时时间戳；
    // 用于插值计算当前激光帧起止时间范围内，每一时刻的旋转姿态
    double *imuTime = new double[queueLength];
    double *imuRotX = new double[queueLength];
    double *imuRotY = new double[queueLength];
    double *imuRotZ = new double[queueLength];

    int imuPointerCur;
    bool firstPointFlag;
    Eigen::Affine3f transStartInverse;//仿射变换矩阵：平移向量+旋转变换组合而成，可以同时实现旋转，缩放，平移等空间变换。

    // 当前帧原始激光点云
    pcl::PointCloud<PointXYZIRT>::Ptr laserCloudIn;
    // 当期帧运动畸变校正之后的激光点云
    pcl::PointCloud<PointType>::Ptr   fullCloud;
    // 从fullCloud中提取有效点
    pcl::PointCloud<PointType>::Ptr   extractedCloud;

    int deskewFlag;//时间戳函数
    cv::Mat rangeMat;//Mat是一个伟大的图像容器类

    bool odomDeskewFlag;
    // 当前激光帧起止时刻对应imu里程计位姿变换，该变换对应的平移增量；用于插值计算当前激光帧起止时间范围内，每一时刻的位置
    float odomIncreX;
    float odomIncreY;
    float odomIncreZ;

    // 当前帧激光点云运动畸变校正之后的数据，包括点云数据、初始位姿、姿态角等，发布给featureExtraction进行特征提取
    lvi_sam::cloud_info cloudInfo;
    // 当前帧起始时刻
    double timeScanCur;
    // 下一帧的开始时刻
    double timeScanNext;
    // 当前帧header，包含时间戳信息
    std_msgs::Header cloudHeader;
        

public:
    ImageProjection():deskewFlag(0)
    {
        // 订阅原始imu数据
         // this: 调用这个class里的返回函数，可以使用第四个参数，例如有个对象叫listener，
         // 调用该对象内部的回调函数，则传入&listener，这里调用自己的，则传入this指针
         // ros::TransportHints().tcpNoDelay() :被用于指定hints，确定传输层的作用话题的方式:无延迟的TCP传输方式

        subImu        = nh.subscribe<sensor_msgs::Imu>        (imuTopic, 2000, &ImageProjection::imuHandler, this, ros::TransportHints().tcpNoDelay());//多线程时使用ros::TransportHints().tcpNoDelay()
        //订阅imu里程计: 来自IMUPreintegration(IMUPreintegration.cpp中的类IMUPreintegration)发布的里程计话题（增量式）
        // 订里程计，由imuPreintegration积分计算得到的每时刻imu位姿
        subOdom       = nh.subscribe<nav_msgs::Odometry>      (PROJECT_NAME + "/vins/odometry/imu_propagate_ros", 2000, &ImageProjection::odometryHandler, this, ros::TransportHints().tcpNoDelay());
        // 订阅原始lidar数据
        subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(pointCloudTopic, 5, &ImageProjection::cloudHandler, this, ros::TransportHints().tcpNoDelay());
       
        // 发布当前激光帧运动畸变校正后的点云，有效点
        pubExtractedCloud = nh.advertise<sensor_msgs::PointCloud2> (PROJECT_NAME + "/lidar/deskew/cloud_deskewed", 5);
         // 发布当前激光帧运动畸变校正后的点云信息
        pubLaserCloudInfo = nh.advertise<lvi_sam::cloud_info>      (PROJECT_NAME + "/lidar/deskew/cloud_info", 5);

         // 初始化 分配内存
        allocateMemory();
        //重置参数
        resetParameters();
        // pcl日志级别，只打ERROR日志
        pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
    }
    
    //初始化，为3个指针分配内存，调用resetParameters()
    void allocateMemory()
    {    //根据params.yaml中给出的N_SCAN Horizon_SCAN参数值分配内存
        //用智能指针的reset方法在构造函数里面进行初始化
        laserCloudIn.reset(new pcl::PointCloud<PointXYZIRT>());//p.reset(q.d); //将p中内置指针换为q，并且用d来释放p之前所指的空间
        fullCloud.reset(new pcl::PointCloud<PointType>());
        extractedCloud.reset(new pcl::PointCloud<PointType>());
        
        fullCloud->points.resize(N_SCAN*Horizon_SCAN);
       
        //cloudinfo是msg文件下自定义的cloud_info消息，对其中的变量进行赋值操作
        //assign(int size, int value):size-要分配的值数,value-要分配给向量名称的值
        cloudInfo.startRingIndex.assign(N_SCAN, 0);//函数assign()常用在给string类变量赋值.
        cloudInfo.endRingIndex.assign(N_SCAN, 0);

        cloudInfo.pointColInd.assign(N_SCAN*Horizon_SCAN, 0);
        cloudInfo.pointRange.assign(N_SCAN*Horizon_SCAN, 0);

        resetParameters();
    }
    
    // 6.重置参数，等待下一次回调函数执行。
    // 重置参数，接收每帧lidar数据都要重置这些参数，全部归零
    void resetParameters()
    {   //清零操作
        laserCloudIn->clear();
        extractedCloud->clear();
        
        //初始全部用FLT_MAX 填充
        //因此后文函数projectPointCloud中有一句if (rangeMat.at<float>(rowIdn, columnIdn) != FLT_MAX) continue;
        // 深度图像的初始化。reset range matrix for range image projection
        rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));//OpenCV可以scalar::all(n)，就是原来的CvScalarAll(n)；

        imuPointerCur = 0;//当前IMU数据指针设为0
        firstPointFlag = true;//判断是否为第一个点
        odomDeskewFlag = false;//odom是否校正

        for (int i = 0; i < queueLength; ++i)
        {
            imuTime[i] = 0;
            imuRotX[i] = 0;
            imuRotY[i] = 0;
            imuRotZ[i] = 0;
        }
    }

    ~ImageProjection(){}

    //简单地接受消息，然后放入imu消息的队列imuQueue中。
    void imuHandler(const sensor_msgs::Imu::ConstPtr& imuMsg)
    {
        // imu原始测量数据转换到lidar系，加速度、角速度、RPY
        //imuConverter()的实现在utility
        sensor_msgs::Imu thisImu = imuConverter(*imuMsg);
        // 上锁，添加数据的时候队列不可用；执行完函数的时候自动解锁
        std::lock_guard<std::mutex> lock1(imuLock);
        imuQueue.push_back(thisImu);
    }

    //接收轮式里程计的信息，得到的消息放入缓存队列里面，等待之后的使用。由imuPreintegration中的积分计算得到的每时刻位姿(地图优化程序中发布的)
    void odometryHandler(const nav_msgs::Odometry::ConstPtr& odometryMsg)
    {
        std::lock_guard<std::mutex> lock2(odoLock);
        odomQueue.push_back(*odometryMsg);
    }

    // 重要功能
    // 重要功能 
    // 重要功能
    //获取当激光雷达的数据，回调函数执行是直接调用一个个写好的类函数。而这整一个流程是imageProjection的关键
    // 1.首先是接受数据，计算时间戳，检查数据。
    // 2.然后与里程计和imu数据进行时间戳同步，并处理
    // 3.接着检查并校准数据。
    // 4.提取信息。
    // 5.发布校准后的数据，以及提取到的信息。
    void cloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg)
    {    //添加一帧激光点云到队列，取出最早一帧作为当前帧，计算起止时间戳，检查数据有效性
        if (!cachePointCloud(laserCloudMsg))
            return;

        if (!deskewInfo())
            return;

        projectPointCloud();

        cloudExtraction();

        publishClouds();

        resetParameters();
    }

    // 1.首先是接受数据，计算时间戳，检查数据。在cachePointCloud()函数，函数的作用是添加一帧激光点云到队列，取出最早一帧作为当前帧，计算起止时间戳，检查数据有效性。
    bool cachePointCloud(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg)
    {
        // cache point cloud
        cloudQueue.push_back(*laserCloudMsg);//将激光点云添加到队列中
        // 队列缓存中数据过少
        if (cloudQueue.size() <= 2)//判断队列中的点云数量是否大于2，如果小于2，则不进行处理
            return false;
        else
        {    // 取出激光点云队列中最早的一帧
            currentCloudMsg = cloudQueue.front();//当前容器起始元素的引用，传到currentCloudMsg
            cloudQueue.pop_front();//删除第一个或最后一个元素pop_front 和pop_back 函数—list可以，queue也可以，vector不支持
            
            // 当前帧头部
            cloudHeader = currentCloudMsg.header;
             // 当前帧起始时刻
            timeScanCur = cloudHeader.stamp.toSec();
            // 下一帧的开始时刻（而不是当前帧的结束时刻）在前面已经删除了最后一个元素，所以这里取出的是下一帧
            timeScanNext = cloudQueue.front().header.stamp.toSec();
        }

        // convert cloud  // 转换成pcl点云格式 形参: (in,out)
        pcl::fromROSMsg(currentCloudMsg, *laserCloudIn);

        // check dense flag  // 存在无效点，Nan或者Inf
        if (laserCloudIn->is_dense == false)
        {
            ROS_ERROR("Point cloud is not in dense format, please remove NaN points first!");//点云格式不密集，请先删除点NAN
            ros::shutdown();
        }

        // check ring channel// 检查是否存在ring通道，注意static只检查一次，检查ring这个field是否存在. veloodyne和ouster都有;
        // Velodyne激光雷达程序里用了点云信息的ring这个field，如果是其他雷达可能没有这个，因此直接报错，
        // 所以如果雷达是其他品牌且没有ring或者类似于直接给出线束的field，建议按线束的角度直接算，可参考lego-loam中原始代码
        static int ringFlag = 0;//ring代表线数，0是最下面那条
        if (ringFlag == 0)
        {
            ringFlag = -1;
            for (int i = 0; i < (int)currentCloudMsg.fields.size(); ++i)
            {
                if (currentCloudMsg.fields[i].name == "ring")
                {
                    ringFlag = 1;
                    break;
                }
            }
            if (ringFlag == -1)
            {
                ROS_ERROR("Point cloud ring channel not available, please configure your point cloud data!");//点云ring通道不可用，请配置您的点云数据！uint16_t ring;线程
                ros::shutdown();
            }
        }     

        // check point time// 检查时间戳，以及是否存在time
        if (deskewFlag == 0)
        {
            deskewFlag = -1;
            for (int i = 0; i < (int)currentCloudMsg.fields.size(); ++i)
            {
                if (currentCloudMsg.fields[i].name == timeField)
                {
                    deskewFlag = 1;
                    break;
                }
            }
            if (deskewFlag == -1)
                ROS_WARN("Point cloud timestamp not available, deskew function disabled, system will drift significantly!");//点云时间戳不可用，删除功能禁用，系统将漂移明显
        }

        return true;
    }

    // 2.然后与里程计和imu数据进行时间戳同步，并处理.deskewInfo()里，用于处理当前激光帧起止时刻对应的IMU数据、odom里程计数据。
    // imuDeskewInfo();用于处理IMU数据。
    // odomDeskewInfo();用于处理IMU里程计数据。
    bool deskewInfo()
    {
        std::lock_guard<std::mutex> lock1(imuLock);
        std::lock_guard<std::mutex> lock2(odoLock);

        // make sure IMU data available for the scan  // 确保IMU数据的时间戳包含了整个lidar数据的时间戳，不能只传输否则就不处理了，
        if (imuQueue.empty() || imuQueue.front().header.stamp.toSec() > timeScanCur || imuQueue.back().header.stamp.toSec() < timeScanNext)
        {
            ROS_DEBUG("Waiting for IMU data ...");
            return false;
        }

        // 当前帧对应imu数据处理
        // 1、遍历当前激光帧起止时刻之间的imu数据，初始时刻对应imu的姿态角RPY设为当前帧的初始姿态角
        // 2、用角速度、时间积分，计算每一时刻相对于初始时刻的旋转量，初始时刻旋转设为0
        // 注：imu数据都已经转换到lidar系下了
        //imu去畸变参数计算 
    	// 注：imu数据都已经转换到lidar系下了
        imuDeskewInfo();

        // 当前帧对应odom里程计处理
        // 注：odom里程计都已经转换到lidar系下了
        odomDeskewInfo();

        return true;
    }
    
    // 2.1
    //修剪imu的数据队列，直到imu的时间处于这帧点云的时间内
    // 遍历当前激光帧起止时刻之间的imu数据，初始时刻对应imu的姿态角RPY，
    //设为当前帧的初始姿态角，然后用角速度、时间积分，计算每一时刻相对于初始时刻的旋转量，初始时刻旋转设为0。
    void imuDeskewInfo()//返回 cloudInfo.imuAvailable = true;
    {  
        
        cloudInfo.imuAvailable = false;//这个参数在地图优化mapOptmization.cpp程序中用到  首先为false 完成相关操作后置true

        while (!imuQueue.empty())
        {
            //本来是满足imuQueue.front().header.stamp.toSec() < timeScanCur 的 
            //继续以0.01为阈值 舍弃较旧的imu数据
            if (imuQueue.front().header.stamp.toSec() < timeScanCur - 0.01)
                imuQueue.pop_front();
            else
                break;
        }

        if (imuQueue.empty())
            return;

        imuPointerCur = 0;

        for (int i = 0; i < (int)imuQueue.size(); ++i)
        {   
            //取出imu队列中的数据
            sensor_msgs::Imu thisImuMsg = imuQueue[i];
            //时间戳
            double currentImuTime = thisImuMsg.header.stamp.toSec();

            // get roll, pitch, and yaw estimation for this scan
            //关键的一点  用imu的欧拉角做扫描的位姿估计  这里直接把值给了cloudInfo在地图优化程序中使用 
             // 提取imu姿态角RPY，作为当前lidar帧初始姿态角
            if (currentImuTime <= timeScanCur)
                imuRPY2rosRPY(&thisImuMsg, &cloudInfo.imuRollInit, &cloudInfo.imuPitchInit, &cloudInfo.imuYawInit);//imuRollInit, imuPitchInit,imuYawInit三个值已经在cloudInfo中定义了直接调用其地址
            
            // 超过当前lidar数据的时间戳结束时刻0.01s，结束
            if (currentImuTime > timeScanNext + 0.01)
                break;

            // 第一帧imu旋转角初始化
            if (imuPointerCur == 0)
            {
                imuRotX[0] = 0;
                imuRotY[0] = 0;
                imuRotZ[0] = 0;
                imuTime[0] = currentImuTime;
                ++imuPointerCur;
                continue;
            }

            //从imu信息中直接获得角速度  调用utility.h中函数
            double angular_x, angular_y, angular_z;
            imuAngular2rosAngular(&thisImuMsg, &angular_x, &angular_y, &angular_z);

            //把角速度和时间间隔积分出转角  用于后续的去畸变
            double timeDiff = currentImuTime - imuTime[imuPointerCur-1];//时间间隔等于当前imu时间队列下一个的时间。
            imuRotX[imuPointerCur] = imuRotX[imuPointerCur-1] + angular_x * timeDiff;//角加速度 * （　当前帧imu的时间 - 上一帧imu的时间　）
            imuRotY[imuPointerCur] = imuRotY[imuPointerCur-1] + angular_y * timeDiff;
            imuRotZ[imuPointerCur] = imuRotZ[imuPointerCur-1] + angular_z * timeDiff;
            imuTime[imuPointerCur] = currentImuTime;
            ++imuPointerCur;
        }

        --imuPointerCur;//不知道为什么加加又减减，可能是去除处理的第一帧的初始化，用于下面的判断

        if (imuPointerCur <= 0)
            return;

        cloudInfo.imuAvailable = true;
    }
   
    // 2.2
    //初始pose信息保存在cloudInfo里
    // 遍历当前激光帧起止时刻之间的odom里程计数据，初始时刻对应odom里程计，设为当前帧的初始位姿，
    //并用起始、终止时刻对应里程计，计算相对位姿变换，求得第一个和最后一个里程计间的平移量，保存平移增量。
    void odomDeskewInfo()
    {
        cloudInfo.odomAvailable = false;//这个参数在地图优化mapOptmization.cpp程序中用到  首先为false 完成相关操作后置true
        //是否为空判定
        // 修剪odom的数据队列，直到odom的时间处于这帧点云的时间前0.01秒内
        while (!odomQueue.empty())
        {
            if (odomQueue.front().header.stamp.toSec() < timeScanCur - 0.01)
                odomQueue.pop_front();
            else
                break;
        }

        if (odomQueue.empty())
            return;

        if (odomQueue.front().header.stamp.toSec() > timeScanCur)
            return;

        // get start odometry at the beinning of the scan,
        // 提取初始帧的odom数据
        nav_msgs::Odometry startOdomMsg;

        //对于 startOdomMsg 时间进行限定
        for (int i = 0; i < (int)odomQueue.size(); ++i)
        {
            startOdomMsg = odomQueue[i];
            //由于之前已经将小于timeScanCur超过0.01的数据推出  所以startOdomMsg已经可代表起始激光扫描的起始时刻的里程计消息
            if (ROS_TIME(&startOdomMsg) < timeScanCur)
                continue;
            else
                break;
        }

        // 提取里程计姿态角
        tf::Quaternion orientation;//使用geometry_msgs::PoseStampedPtr pose构造一个对象
        tf::quaternionMsgToTF(startOdomMsg.pose.pose.orientation, orientation);//把geometry_msg形式的四元数转化为tf形式四元数

        // 在ROS中，没有直接的四元数转欧拉角的函数方法,先将四元数转换成了旋转矩阵tf::Matrix3x3,再用tf::Matrix3x3内部的getRPY方法来将旋转矩阵转换成欧拉角
        double roll, pitch, yaw;
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);//getRPY来将四元数转换成欧拉角
        

        // Initial guess used in mapOptimization
        // 用当前激光帧起始时刻的odom，初始化lidar位姿，后面用于mapOptmization
        cloudInfo.odomX = startOdomMsg.pose.pose.position.x;
        cloudInfo.odomY = startOdomMsg.pose.pose.position.y;
        cloudInfo.odomZ = startOdomMsg.pose.pose.position.z;
        cloudInfo.odomRoll  = roll;
        cloudInfo.odomPitch = pitch;
        cloudInfo.odomYaw   = yaw;
        cloudInfo.odomResetId = (int)round(startOdomMsg.pose.covariance[0]);

        cloudInfo.odomAvailable = true;  //标志变量 

        // get end odometry at the end of the scan//获得一帧扫描末尾的里程计消息  这个就跟初始位姿估计没有关系 只是用于去畸变 运动补偿
        odomDeskewFlag = false;
        //相关时间先后关系的判断  odom队列中最后的数据大于下一帧扫描的时间
        //odom最后一个数据的时间比雷达结束的时间早，不能覆盖点云的时间
        if (odomQueue.back().header.stamp.toSec() < timeScanNext)
            return;
        //获取第一个比雷达结束的时间晚的odom
        nav_msgs::Odometry endOdomMsg;

        //对于endOdomMsg的时间进行一个限定
        for (int i = 0; i < (int)odomQueue.size(); ++i)
        {
            endOdomMsg = odomQueue[i];
            // 在cloudHandler的cachePointCloud函数中
            //timeScanEnd = timeScanCur + laserCloudIn->points.back().time;
            // 找到第一个大于一帧激光结束时刻的odom           
            if (ROS_TIME(&endOdomMsg) < timeScanNext)
                continue;
            else
                break;
        }
        
        // 如果起止时刻对应odom的协方差不等，可能是里程计出现了问题，它们两个的相关性应该是一致的，所以直接返回
        if (int(round(startOdomMsg.pose.covariance[0])) != int(round(endOdomMsg.pose.covariance[0])))
            return;
        //获得起始的变换
        Eigen::Affine3f transBegin = pcl::getTransformation(startOdomMsg.pose.pose.position.x, startOdomMsg.pose.pose.position.y, startOdomMsg.pose.pose.position.z, roll, pitch, yaw);

        //获得结束时的变换
        tf::quaternionMsgToTF(endOdomMsg.pose.pose.orientation, orientation);
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
        Eigen::Affine3f transEnd = pcl::getTransformation(endOdomMsg.pose.pose.position.x, endOdomMsg.pose.pose.position.y, endOdomMsg.pose.pose.position.z, roll, pitch, yaw);
        
        //获得一帧扫描起始与结束时刻间的变换  这个在loam里解释的比较清楚
        Eigen::Affine3f transBt = transBegin.inverse() * transEnd;

        float rollIncre, pitchIncre, yawIncre;
        pcl::getTranslationAndEulerAngles(transBt, odomIncreX, odomIncreY, odomIncreZ, rollIncre, pitchIncre, yawIncre);

        odomDeskewFlag = true;
    }
    
    //3.1.1
    //工具性质的函数
    // 根据点云中某点的时间戳赋予其对应插值得到的旋转量
    void findRotation(double pointTime, float *rotXCur, float *rotYCur, float *rotZCur)
    {
        *rotXCur = 0; *rotYCur = 0; *rotZCur = 0;
        // 查找当前时刻在imuTime下的索引
        int imuPointerFront = 0;
         //imuDeskewInfo中，对imuPointerCur进行计数(计数到超过当前激光帧结束时刻0.01s)
        while (imuPointerFront < imuPointerCur)
        {
            //imuTime在imuDeskewInfo（deskewInfo中调用，deskewInfo在cloudHandler中调用）被赋值，从imuQueue中取值
            //pointTime为当前时刻，由此函数的函数形参传入,要找到imu积分列表里第一个大于当前时间的索引
            if (pointTime < imuTime[imuPointerFront])
                break;
            ++imuPointerFront;
        }
        // 设为离当前时刻最近的旋转增量
        // 如果上边的循环没进去或者到了最大执行次数，则只能近似的将当前的旋转赋值过去
        if (pointTime > imuTime[imuPointerFront] || imuPointerFront == 0)
        {
            //未找到大于当前时刻的imu积分索引
            //imuRotX等为之前积分出的内容.(imuDeskewInfo中)
            *rotXCur = imuRotX[imuPointerFront];
            *rotYCur = imuRotY[imuPointerFront];
            *rotZCur = imuRotZ[imuPointerFront];
        } 
        else
        {
            // 前后时刻插值计算当前时刻的旋转增量
            //根据点的时间信息 获得每个点的时刻的旋转变化量            
            //此时front的时间是大于当前pointTime时间，back=front-1刚好小于当前pointTime时间，前后时刻插值计算

            int imuPointerBack = imuPointerFront - 1;
            double ratioFront = (pointTime - imuTime[imuPointerBack]) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            double ratioBack = (imuTime[imuPointerFront] - pointTime) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            // 按前后百分比赋予旋转量
            *rotXCur = imuRotX[imuPointerFront] * ratioFront + imuRotX[imuPointerBack] * ratioBack;
            *rotYCur = imuRotY[imuPointerFront] * ratioFront + imuRotY[imuPointerBack] * ratioBack;
            *rotZCur = imuRotZ[imuPointerFront] * ratioFront + imuRotZ[imuPointerBack] * ratioBack;
        }
    }
    
    //3.1.2
    //工具性质的函数
    void findPosition(double relTime, float *posXCur, float *posYCur, float *posZCur)
    {
        *posXCur = 0; *posYCur = 0; *posZCur = 0;

        // if (cloudInfo.odomAvailable == false || odomDeskewFlag == false)
        //     return;

        // float ratio = relTime / (timeScanNext - timeScanCur);

        // *posXCur = ratio * odomIncreX;
        // *posYCur = ratio * odomIncreY;
        // *posZCur = ratio * odomIncreZ;
    }
    
    //3.1激光运动畸变校正
    //用当前帧起止时刻之间的imu数据计算旋转增量，里程计数据计算平移增量，
    //进而将每一时刻激光点位置变换到第一个激光点坐标系下，对点云的每个点进行畸变校正·
    PointType deskewPoint(PointType *point, double relTime)//relTime:laserCloudIn->points[i].time
    {
        // 检查是否有时间戳信息，和是否有合规的imu数据
        if (deskewFlag == -1 || cloudInfo.imuAvailable == false)
            return *point;
        //点的时间等于scan时间加relTime（后文的laserCloudIn->points[i].time）
        //lasercloudin中存储的time是一帧中距离起始点的相对时间
        // 在cloudHandler的cachePointCloud函数中，timeScanCur = cloudHeader.stamp.toSec();，即当前帧点云的初始时刻
        //二者相加即可得到当前点的准确时刻
        double pointTime = timeScanCur + relTime;
        //定义相关变量用于补偿 旋转 平移 
        float rotXCur, rotYCur, rotZCur;
        //调用本程序文件内函数 获得相关旋转量 具体看该函数
        findRotation(pointTime, &rotXCur, &rotYCur, &rotZCur);

        float posXCur, posYCur, posZCur;
        //调用本程序文件函数 获得相关平移量
        findPosition(relTime, &posXCur, &posYCur, &posZCur);

        //如果是第一此收到数据
        //这里的firstPointFlag来源于resetParameters函数，
        //而resetParameters函数每次ros调用cloudHandler都会启动  第一个点的位姿增量（0），求逆
        if (firstPointFlag == true)
        {
            transStartInverse = (pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur)).inverse();
            firstPointFlag = false;
        }

        // transform points to start，当前时刻激光点与第一个激光点的位姿变换
        //通过pcl::getTransformation获取从雷达坐标系到大地坐标系的旋转矩阵。
        Eigen::Affine3f transFinal = pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur);
        
        //扫描该点相对扫描本次scan第一个点的lidar变换矩阵
         //第一个点时lidar世界坐标系下变换矩阵的逆×当前点时lidar世界坐标系下变换矩阵
         //Tij=Twi^-1 * Twj
         //注：这里准确的来说，不是世界坐标系
         //根据代码来看，是把imu积分
         //从imuDeskewInfo函数中，在当前激光帧开始的前0.01秒的imu数据开始积分
         //把它作为原点，然后获取当前激光帧第一个点时刻的位姿增量transStartInverse，
         //和当前点时刻的位姿增量transFinal，根据逆矩阵计算二者变换transBt。
         //因此相对的不是“世界坐标系”,
         //而是“当前激光帧开始前的0.01秒的雷达坐标系（在imucallback函数中已经把imu转换到了雷达坐标系了）
        Eigen::Affine3f transBt = transStartInverse * transFinal;
        //当前激光点在第一个激光点坐标系下的坐标
        PointType newPoint;
        newPoint.x = transBt(0,0) * point->x + transBt(0,1) * point->y + transBt(0,2) * point->z + transBt(0,3);
        newPoint.y = transBt(1,0) * point->x + transBt(1,1) * point->y + transBt(1,2) * point->z + transBt(1,3);
        newPoint.z = transBt(2,0) * point->x + transBt(2,1) * point->y + transBt(2,2) * point->z + transBt(2,3);
        newPoint.intensity = point->intensity;

        return newPoint;
    }

    // 3.接着检查并校准数据。
    void projectPointCloud()
    {
        int cloudSize = (int)laserCloudIn->points.size();
        // range image projection// 遍历当前帧激光点云
        for (int i = 0; i < cloudSize; ++i)
        {
            // pcl格式
            PointType thisPoint;
            //laserCloudIn就是原始的点云话题中的数据
            thisPoint.x = laserCloudIn->points[i].x;
            thisPoint.y = laserCloudIn->points[i].y;
            thisPoint.z = laserCloudIn->points[i].z;
            thisPoint.intensity = laserCloudIn->points[i].intensity;
            
            //距离图像的行  与点云中ring对应,
            //rowIdn计算出该点激光雷达是水平方向上第几线的。从下往上计数，-15度记为初始线，第0线，一共16线(N_SCAN=16
            int rowIdn = laserCloudIn->points[i].ring;
            if (rowIdn < 0 || rowIdn >= N_SCAN)
                continue;

            // 扫描线如果有降采样，跳过采样的扫描线这里要跳过
            if (rowIdn % downsampleRate != 0)
                continue;
           
            // 水平扫描角度
            float horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;
            
            // 水平扫描角度步长，例如一周扫描1800次，则两次扫描间隔角度0.2°
            //角分辨率 360/1800
            static float ang_res_x = 360.0/float(Horizon_SCAN);
            //计算在距离图像上点属于哪一列
            //horizonAngle 为[-180,180],horizonAngle -90 为[-270,90],-round 为[-90,270], /ang_res_x 为[-450,1350]
            //+Horizon_SCAN/2为[450,2250]
            // 即把horizonAngle从[-180,180]映射到[450,2250]
            int columnIdn = -round((horizonAngle-90.0)/ang_res_x) + Horizon_SCAN/2;
                //大于1800，则减去1800，相当于把1801～2250映射到1～450
                //先把columnIdn从horizonAngle:(-PI,PI]转换到columnIdn:[H/4,5H/4],
                //然后判断columnIdn大小，把H到5H/4的部分切下来，补到0～H/4的部分。
                //将它的范围转换到了[0,H] (H:Horizon_SCAN)。
                //这样就把扫描开始的地方角度为0与角度为360的连在了一起，非常巧妙。
                //如果前方是x，左侧是y，那么正后方左边是180，右边是-180。这里的操作就是，把它展开成一幅图:
            if (columnIdn >= Horizon_SCAN)
                columnIdn -= Horizon_SCAN;

            if (columnIdn < 0 || columnIdn >= Horizon_SCAN)
                continue;
            //调用utility.h里的函数计算点到雷达的距离
            float range = pointDistance(thisPoint);
            
            // 如果距离小于一个阈值，则跳过该点，比如说扫到手持设备的人
            if (range < 1.0)
                continue;
            // opencv之遍历
            // 使用at成员函数
            // 已经存过该点，不再处理
            if (rangeMat.at<float>(rowIdn, columnIdn) != FLT_MAX)
                continue;

            // for the amsterdam dataset对于指定数据集
            // if (range < 6.0 && rowIdn <= 7 && (columnIdn >= 1600 || columnIdn <= 200))
            //     continue;
            // if (thisPoint.z < -2.0)
            //     continue;

            // 矩阵存激光点的距离
            rangeMat.at<float>(rowIdn, columnIdn) = range;
            // 激光运动畸变校正
            // 利用当前帧起止时刻之间的imu数据计算旋转增量，odom里程计数据计算平移增量，
            // 进而将每一时刻激光点位置变换到第一个激光点坐标系下，进行运动补偿
            thisPoint = deskewPoint(&thisPoint, laserCloudIn->points[i].time); // Velodyne
            // thisPoint = deskewPoint(&thisPoint, (float)laserCloudIn->points[i].t / 1000000000.0); // Ouster
            // 转换成一维索引，存校正之后的激光点
            int index = columnIdn  + rowIdn * Horizon_SCAN;
            fullCloud->points[index] = thisPoint;
        }
    }

    // 4.提取信息。提取有效激光点，存extractedCloud。
    void cloudExtraction()
    {
        // 有效激光点数量
        int count = 0;
        // extract segmented cloud for lidar odometry
        // 为激光雷达测距提取分割的云层，遍历所有点
        for (int i = 0; i < N_SCAN; ++i)
        {
            // 记录每根扫描线起始第5个激光点在一维数组中的索引
            cloudInfo.startRingIndex[i] = count - 1 + 5;

            for (int j = 0; j < Horizon_SCAN; ++j)
            {
                 // 有效激光点
                if (rangeMat.at<float>(i,j) != FLT_MAX)
                {
                    // 记录激光点对应的Horizon_SCAN方向上的索引。
                    // mark the points' column index for marking occlusion later
                    cloudInfo.pointColInd[count] = j;
                    // save range info
                    // 保存范围信息
                    cloudInfo.pointRange[count] = rangeMat.at<float>(i,j);
                    // save extracted cloud
                    // 保存提取出来的点云
                    extractedCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                    // size of extracted cloud
                    // 提取点云的大小
                    ++count;
                }
            }
            // 记录每根扫描线倒数第5个激光点在一维数组中的索引
            cloudInfo.endRingIndex[i] = count -1 - 5;
        }
    }
    
    // 5.发布校准后的数据，以及提取到的信息。当前帧校正后点云，有效点。
    void publishClouds()
    {
        cloudInfo.header = cloudHeader;
        //publishCloud在utility.h头文件中,需要传入发布句柄pubExtractedCloud，提取出的有效点云，该帧时间戳，
        //pubExtractedCloud定义在构造函数中，用来发布去畸变的点云.
        //extractedCloud主要在cloudExtraction中被提取，点云被去除了畸变，//另外每行头五个和后五个不要(（仍然被保存，但是之后在提取特征时不要,因为要根据前后五个点算曲率）
        //cloudHeader.stamp 来源于currentCloudMsg,cloudHeader在cachePointCloud中被赋值currentCloudMsg.header //而currentCloudMsg是点云队列cloudQueue中提取的
        //lidarFrame:在utility.h中被赋为base_link,
        //在publishCloud函数中，tempCloud.header.frame_id="base_link"(lidarFrame)//之后用发布句柄pubExtractedCloud来发布去畸变的点云

        cloudInfo.cloud_deskewed  = publishCloud(&pubExtractedCloud, extractedCloud, cloudHeader.stamp, "base_link");
        //pubExtractedCloud发布的只有点云信息，而pubLaserCloudInfo发布的为自定义的很多信息
        //发布自定义cloud_info信息
        pubLaserCloudInfo.publish(cloudInfo);
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "lidar");

    ImageProjection IP;
    
    ROS_INFO("\033[1;32m----> Lidar Cloud Deskew Started.\033[0m");
    //对于一些只订阅一个话题的简单节点来说，我们使用ros::spin()进入接收循环，
    //每当有订阅的话题发布时，进入回调函数接收和处理消息数据。
    //但是更多的时候，一个节点往往要接收和处理不同来源的数据，并且这些数据的产生频率也各不相同，
    //当我们在一个回调函数里耗费太多时间时，会导致其他回调函数被阻塞，导致数据丢失。
    //这种场合需要给一个节点开辟多个线程，保证数据流的畅通。
    ros::MultiThreadedSpinner spinner(3);
    spinner.spin();
    
    return 0;
}