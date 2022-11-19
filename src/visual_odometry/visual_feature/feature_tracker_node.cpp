#include "feature_tracker.h"

#define SHOW_UNDISTORTION 0


// mtx lock for two threads
std::mutex mtx_lidar;

// global variable for saving the depthCloud shared between two threads
pcl::PointCloud<PointType>::Ptr depthCloud(new pcl::PointCloud<PointType>());

// global variables saving the lidar point cloud
deque<pcl::PointCloud<PointType>> cloudQueue;
deque<double> timeQueue;

// global depth register for obtaining depth of a feature
DepthRegister *depthRegister;

// feature publisher for VINS estimator
ros::Publisher pub_feature;
ros::Publisher pub_match;
ros::Publisher pub_restart;

// feature tracker variables特征跟踪变量
FeatureTracker trackerData[NUM_OF_CAM];
double first_image_time;//每隔delta_t = 1/FREQ 时间的帧对应的时间戳 
int pub_count = 1;//每隔delta_t = 1/FREQ 时间内连续(没有中断/没有报错)发布的帧数
bool first_image_flag = true;// 0:当前是第一帧   1:当前不是第一帧
double last_image_time = 0;//当前帧或上一帧的时间戳 
bool init_pub = 0;//  0:第一帧不把特征发布到buf里    1:发布到buf里   


//特征提取的回调函数,处理图像，主要包含readImage()和get_depth()   
//拿到的是ros的图像，一定要转换为opencv的图像
void img_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    double cur_img_time = img_msg->header.stamp.toSec();//获取当前图像的时间戳
    //一、如果出现间断，则重新初始化
    
    //首先判断是不是第一帧，如果是，则把数据初始化，记录第一个图像帧的时间。
    if(first_image_flag)
    {
        first_image_flag = false;
        first_image_time = cur_img_time;
        last_image_time = cur_img_time;
        return;
    }

    //二、 detect unstable camera stream频率控制,检测不稳定的相机流
    if (cur_img_time - last_image_time > 1.0 || cur_img_time < last_image_time)
    {
        ROS_WARN("image discontinue! reset the feature tracker!");
        // 重置
        first_image_flag = true; 
        last_image_time = 0;
        pub_count = 1;
        std_msgs::Bool restart_flag;
        restart_flag.data = true;
        pub_restart.publish(restart_flag);
        return;
    }
    last_image_time = cur_img_time;

    //三、控制发布频率(特征点发布频率为FREQ，设置为0-->100，跟相机发布频率一样，最小10Hz）
    //数据准备,需要控制发布频率，以及转换图像编码。
    //a. 控制发布频率 frequency controla. 对于计算机算力的调控
    if (round(1.0 * pub_count / (cur_img_time - first_image_time)) <= FREQ)
    {
        // 是否需要发布特征点的标志
        PUB_THIS_FRAME = true;
        // reset the frequency control,累计发布数量 / 当前收到图片的时间距离设定的首张图片的时间为 小于20HZ（可以调整）
        //pub_count: # 发布图像的个数.FREQ: 20 # 控制图像光流跟踪的频率，这里作者在参数配置文件中设为20HZ
        if (abs(1.0 * pub_count / (cur_img_time - first_image_time) - FREQ) < 0.01 * FREQ)
        {
            first_image_time = cur_img_time;
            pub_count = 0;
        }
    }
    else
    {
        PUB_THIS_FRAME = false;
    }




    //四、将图像编码8UC1转换为mono8重点
    //b. 图像的格式调整和图像读取
    cv_bridge::CvImageConstPtr ptr;//cv_bridge 的toCVCopy函数将ROS图像消息转化为OpenCV图像消息
    if (img_msg->encoding == "8UC1")//8uc1是8bit的单色灰度图。而mono8就是一个8uc1的格式.在我们要修改数据的地方。我们必须复制一份ros的信息数据。
    {
        sensor_msgs::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;//??
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    }
    else
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

     //五、读取图像数据进行处理 ，这里已经拿到了opencv格式的图像数据   

    //img_msg或img都是sensor_msg格式的，我们需要一个桥梁，转换为CV::Mat格式的数据，以供后续图像处理
    cv::Mat show_img = ptr->image;


   // 2. 对最新帧特征点的提取和光流追踪 (核心函数)
    TicToc t_r; // 计算时间的类
    for (int i = 0; i < NUM_OF_CAM; i++)//
    {   
        ROS_DEBUG("processing camera %d", i);
        if (i != 1 || !STEREO_TRACK)
            trackerData[i].readImage(ptr->image.rowRange(ROW * i, ROW * (i + 1)), cur_img_time);//为矩阵的指定行区间创建一个矩阵头。
        else
        {
            if (EQUALIZE)//EQUALIZE是如果光太亮或太暗则为1，用来进行直方图均衡化。
            {
                cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
                clahe->apply(ptr->image.rowRange(ROW * i, ROW * (i + 1)), trackerData[i].cur_img);
            }
            else
                trackerData[i].cur_img = ptr->image.rowRange(ROW * i, ROW * (i + 1));
        }

        #if SHOW_UNDISTORTION
            trackerData[i].showUndistortion("undistrotion_" + std::to_string(i));
        #endif
    }
     
    //六、更新全局ID，updataID函数实现给新特征点对应的ids编号
    //. 对新加入的特征点更新全局id
    for (unsigned int i = 0;; i++)
    {
        bool completed = false;//如果是true，说明没有更新完id，则持续循环，如果是false，说明更新完了则跳出循环
        for (int j = 0; j < NUM_OF_CAM; j++)
            if (j != 1 || !STEREO_TRACK)
                completed |= trackerData[j].updateID(i);
        if (!completed)
            break;
    }

   //4. 矫正、封装并发布特征点到pub_feature
   //封装成sensor_msgs::PointCloudPtr类型的feature_points实例中,发布到pub_img
   //pub当前fream
   //如果PUB_THIS_FRAME=1则进行发布将特征点（矫正后归一化平面的3D点(x,y,z=1))，
   //像素2D点(u,v)，像素的速度(vx,vy)，封装成sensor_msgs::PointCloudPtr类型的feature_points实例中,发布到pub_img;
   //将图像封装到cv_bridge::cvtColor类型的ptr实例中发布到pub_match
   if (PUB_THIS_FRAME)
   {
        pub_count++;//发布数量+1
        sensor_msgs::PointCloudPtr feature_points(new sensor_msgs::PointCloud);//新建容器feature_points
        sensor_msgs::ChannelFloat32 id_of_point;//像素编号
        sensor_msgs::ChannelFloat32 u_of_point;// 像素坐标x
        sensor_msgs::ChannelFloat32 v_of_point;// 像素坐标y
        sensor_msgs::ChannelFloat32 velocity_x_of_point;//像素速度x
        sensor_msgs::ChannelFloat32 velocity_y_of_point;//像素速度y

        feature_points->header.stamp = img_msg->header.stamp;
        feature_points->header.frame_id = "vins_body";

        vector<set<int>> hash_ids(NUM_OF_CAM);// 哈希表id
        for (int i = 0; i < NUM_OF_CAM; i++) // 循环相机数量
        {
            auto &un_pts = trackerData[i].cur_un_pts;
            auto &cur_pts = trackerData[i].cur_pts;
            auto &ids = trackerData[i].ids;
            auto &pts_velocity = trackerData[i].pts_velocity;// 特征点的数量
            for (unsigned int j = 0; j < ids.size(); j++)
            {
                if (trackerData[i].track_cnt[j] > 1)// 只发布追踪次数大于1的特征点
                {
                    int p_id = ids[j];
                    hash_ids[i].insert(p_id);//归一化了
                    geometry_msgs::Point32 p;
                    p.x = un_pts[j].x;
                    p.y = un_pts[j].y;
                    p.z = 1;

                    feature_points->points.push_back(p);
                    id_of_point.values.push_back(p_id * NUM_OF_CAM + i);
                    u_of_point.values.push_back(cur_pts[j].x);
                    v_of_point.values.push_back(cur_pts[j].y);
                    velocity_x_of_point.values.push_back(pts_velocity[j].x);
                    velocity_y_of_point.values.push_back(pts_velocity[j].y);
                }
            }
        }
        //对应通道// 封装信息，准备发布
        feature_points->channels.push_back(id_of_point);
        feature_points->channels.push_back(u_of_point);
        feature_points->channels.push_back(v_of_point);
        feature_points->channels.push_back(velocity_x_of_point);
        feature_points->channels.push_back(velocity_y_of_point);

        // get feature depth from lidar point cloud// 从共享内存中获得深度信息
        pcl::PointCloud<PointType>::Ptr depth_cloud_temp(new pcl::PointCloud<PointType>());
        mtx_lidar.lock();
        *depth_cloud_temp = *depthCloud;
        mtx_lidar.unlock();




        // 获取深度信息get_depth      {核心函数}{核心函数}{核心函数}{核心函数}{核心函数}
        sensor_msgs::ChannelFloat32 depth_of_points = depthRegister->get_depth(img_msg->header.stamp, show_img, depth_cloud_temp, trackerData[0].m_camera, feature_points->points);
        feature_points->channels.push_back(depth_of_points);
        



         //八、发布特征点，跳过第一帧图像，第一帧没有光流 
        // skip the first image; since no optical speed on frist image// 第一帧不发布，因为没有光流速度。
        if (!init_pub)
        {
            init_pub = 1;
        }
        else
            pub_feature.publish(feature_points);

        //九、把特征点画在图像中并发布（有深度的绿色，无深度的红色）
        // 封装并发布到pub_match 将图像封装到cv_bridge::cvtColor类型的ptr实例中发布到pub_match
        // publish features in image
        if (pub_match.getNumSubscribers() != 0)
        {
            ptr = cv_bridge::cvtColor(ptr, sensor_msgs::image_encodings::RGB8);
            //cv::Mat stereo_img(ROW * NUM_OF_CAM, COL, CV_8UC3);
            cv::Mat stereo_img = ptr->image;

            for (int i = 0; i < NUM_OF_CAM; i++)
            {
                cv::Mat tmp_img = stereo_img.rowRange(i * ROW, (i + 1) * ROW);
                cv::cvtColor(show_img, tmp_img, CV_GRAY2RGB);// show_img灰度图转RGB（tmp_img）
                 //显示追踪状态，越红越好，越蓝越不行---cv::Scalar决定的
                for (unsigned int j = 0; j < trackerData[i].cur_pts.size(); j++)
                {
                    if (SHOW_TRACK)
                    {
                        // track count // 计算跟踪的特征点数量
                        double len = std::min(1.0, 1.0 * trackerData[i].track_cnt[j] / WINDOW_SIZE);
                        cv::circle(tmp_img, trackerData[i].cur_pts[j], 4, cv::Scalar(255 * (1 - len), 255 * len, 0), 4);
                    } else {
                        // depth  // 结合深度进行计算
                        if(j < depth_of_points.values.size())
                        {
                            if (depth_of_points.values[j] > 0)
                                cv::circle(tmp_img, trackerData[i].cur_pts[j], 4, cv::Scalar(0, 255, 0), 4);
                            else
                                cv::circle(tmp_img, trackerData[i].cur_pts[j], 4, cv::Scalar(0, 0, 255), 4);
                        }
                    }
                }
            }

            pub_match.publish(ptr->toImageMsg());
        }
    }
}


void lidar_callback(const sensor_msgs::PointCloud2ConstPtr& laser_msg)
{
    //减少数据的大小，提高处理的速度,作者每4个就会跳过一次处理lidar_skip: 3 # skip this amount of scans
    static int lidar_count = -1;
    if (++lidar_count % (LIDAR_SKIP+1) != 0)
        return;//订阅雷达目的是给视觉特征点提供深度信息，短时间内图像和雷达变化不大，适当降低雷达频率

    // 0. listen to transform从laser_msg中，获取位姿的信息 // 0、监听相机到W系转换坐标系
    static tf::TransformListener listener;
    static tf::StampedTransform transform;
    //try/catch 语句用于处理代码中可能出现的错误信息。
    try{
        listener.waitForTransform("vins_world", "vins_body_ros", laser_msg->header.stamp, ros::Duration(0.01));//等待tf变换
        listener.lookupTransform("vins_world", "vins_body_ros", laser_msg->header.stamp, transform);//世界坐标系到body坐标系下
    } 
    catch (tf::TransformException ex){
        // ROS_ERROR("lidar no tf");//如果监听不到两个坐标系，不处理雷达，不求深度
        return;
    }

    //0.从获取到的位姿信息中，提取x，y，z和旋转角的信息。然后通过旋转角获取RPY的三个角度，最后，在通过这6个信息，去获得当前位姿的变换矩阵。
    double xCur, yCur, zCur, rollCur, pitchCur, yawCur;
    //平移
    xCur = transform.getOrigin().x();
    yCur = transform.getOrigin().y();
    zCur = transform.getOrigin().z();
    //旋转
    tf::Matrix3x3 m(transform.getRotation());
    m.getRPY(rollCur, pitchCur, yawCur);
    Eigen::Affine3f transNow = pcl::getTransformation(xCur, yCur, zCur, rollCur, pitchCur, yawCur);//构造变换

    // 1. convert laser cloud message to pcl先创建一个pcl格式的容器,通过pcl库中内置的pcl::fromROSMsg的方法，去转换格式
    pcl::PointCloud<PointType>::Ptr laser_cloud_in(new pcl::PointCloud<PointType>());
    pcl::fromROSMsg(*laser_msg, *laser_cloud_in);

    // 2. downsample new cloud (save memory)调用PCL的降采样算法
    // 生成新的PCL格式的点云容器，并调用PCL的降采样算法
    pcl::PointCloud<PointType>::Ptr laser_cloud_in_ds(new pcl::PointCloud<PointType>());
    // 生成过滤器对象，并设置参数
    static pcl::VoxelGrid<PointType> downSizeFilter;
    // 设置过滤的大小，设置采样体素大小
    downSizeFilter.setLeafSize(0.2, 0.2, 0.2);//体素尺寸通过函数setLeafSize()设置
    downSizeFilter.setInputCloud(laser_cloud_in);//待处理的点云通过函数setInputCloud导入体素对象
    downSizeFilter.filter(*laser_cloud_in_ds);//函数filter (*cloud_filtered)设置了降采样结果的保存位置
    *laser_cloud_in = *laser_cloud_in_ds;// 把过滤好的数据覆盖原本的数据

    // 3. filter lidar points (only keep points in camera view)保证当前点云的点在当前相机视角内??
    // 生成新的PCL格式的点云容器.滤除雷达点云，只保留相机视野部分
    pcl::PointCloud<PointType>::Ptr laser_cloud_in_filter(new pcl::PointCloud<PointType>());
    // 遍历所有的点
    for (int i = 0; i < (int)laser_cloud_in->size(); ++i)
    {
        PointType p = laser_cloud_in->points[i];
        // 符合条件的数据，放入laser_cloud_in_filter,以x轴的光锥
        if (p.x >= 0 && abs(p.y / p.x) <= 10 && abs(p.z / p.x) <= 10)
            laser_cloud_in_filter->push_back(p);
    }
    *laser_cloud_in = *laser_cloud_in_filter;

    // TODO: transform to IMU body frame将点云从激光雷达坐标系变成相机坐标系
    // 4. offset T_lidar -> T_camera 雷达点云转换（雷达系->相机系） offset T_lidar -> T_camera 
    pcl::PointCloud<PointType>::Ptr laser_cloud_offset(new pcl::PointCloud<PointType>());
    // 获得tf变换信息，生成变换矩阵。
    Eigen::Affine3f transOffset = pcl::getTransformation(L_C_TX, L_C_TY, L_C_TZ, L_C_RX, L_C_RY, L_C_RZ);//读取参数：L_C_TX, L_C_TY, L_C_TZ, L_C_RX, L_C_RY, L_C_RZ
    // 从lidar坐标系转变到相机坐标系
    pcl::transformPointCloud(*laser_cloud_in, *laser_cloud_offset, transOffset);
    *laser_cloud_in = *laser_cloud_offset;

    // 5. transform new cloud into global odom frame再把点云变换到世界坐标系(静态)，雷达点云转换（相机系->W系）
    pcl::PointCloud<PointType>::Ptr laser_cloud_global(new pcl::PointCloud<PointType>());
    pcl::transformPointCloud(*laser_cloud_in, *laser_cloud_global, transNow);

    // 6. save new cloud把变换完成的点云存储到待处理队列，保存新的全局系点云
    double timeScanCur = laser_msg->header.stamp.toSec();
    cloudQueue.push_back(*laser_cloud_global);
    timeQueue.push_back(timeScanCur);

    // 7. pop old cloud保持队列的时效，剔除五秒之前的点云，给图像特征点深度信息，5S点云信息足够
    while (!timeQueue.empty())
    {
        if (timeScanCur - timeQueue.front() > 5.0)
        {
            cloudQueue.pop_front();
            timeQueue.pop_front();
        } else {
            break;
        }
    }

    std::lock_guard<std::mutex> lock(mtx_lidar);// 先拿到共享内存的锁，


    // 8. fuse global cloud将队列里的点云输入作为总体的待处理深度图。把depthCloud深度点云图先初始化为空，然后将刚刚处理的数据，放入其中
    //把队列的点云放到depthCloud里面fuse global cloud
    depthCloud->clear();
    for (int i = 0; i < (int)cloudQueue.size(); ++i)
        *depthCloud += cloudQueue[i];

    // 9. downsample global cloud降采样总体的深度图
    pcl::PointCloud<PointType>::Ptr depthCloudDS(new pcl::PointCloud<PointType>());
    downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
    downSizeFilter.setInputCloud(depthCloud);
    downSizeFilter.filter(*depthCloudDS);
    *depthCloud = *depthCloudDS;
}

int main(int argc, char **argv)
{
    // initialize ROS node//一、初始化ROS节点
    ros::init(argc, argv, "vins");
    ros::NodeHandle n;
    ROS_INFO("\033[1;32m----> Visual Feature Tracker Started.\033[0m");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Warn);
    readParameters(n);//使用readParameters()函数读入这个节点所需要的参数,文件Parameters.cpp

    // read camera params读入相机接口参数
    for (int i = 0; i < NUM_OF_CAM; i++)
        trackerData[i].readIntrinsicParameter(CAM_NAMES[i]);//FeatureTracker trackerData[NUM_OF_CAM];已定义

    // load fisheye mask to remove features on the boundry加载鱼眼mask来去除边界上的特征。
    if(FISHEYE)
    {
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            trackerData[i].fisheye_mask = cv::imread(FISHEYE_MASK, 0);
            if(!trackerData[i].fisheye_mask.data)
            {
                ROS_ERROR("load fisheye mask fail");
                ROS_BREAK();
            }
            else
                ROS_INFO("load mask success");
        }
    }

    // initialize depthRegister (after readParameters())初始化depthRegister类深度注册类，读入参数后初始化
    depthRegister = new DepthRegister(n);
    
    // subscriber to image and lidar// 订阅原始的图像和世界坐标系下的lidar点云
    //四、订阅图像和雷达topic，两个主线程
    ros::Subscriber sub_img   = n.subscribe(IMAGE_TOPIC,       5,    img_callback);
   
    ros::Subscriber sub_lidar = n.subscribe(POINT_CLOUD_TOPIC, 5,    lidar_callback);
    if (!USE_LIDAR)
        sub_lidar.shutdown();//可选不使用雷达

    // messages to vins estimator// 给vins estimator的消息
    //五、发布3D视觉特征点、带特征点的图像、重启标志
    pub_feature = n.advertise<sensor_msgs::PointCloud>(PROJECT_NAME + "/vins/feature/feature",     5);
    
    pub_match   = n.advertise<sensor_msgs::Image>     (PROJECT_NAME + "/vins/feature/feature_img", 5);
    //即跟踪的特征点信息，由/vins_estimator订阅并进行优化 
    
    pub_restart = n.advertise<std_msgs::Bool>         (PROJECT_NAME + "/vins/feature/restart",     5);
     //判断特征跟踪模块是否出错，若有问题则进行复位，由/vins_estimator订阅

    // two ROS spinners for parallel processing (image and lidar)用ROS的多线程机制，申请2个多线程来运行当有图片消息和lidar消息到来的时候的回调函数
    //六、双线程分别处理img_callback和lidar_callback
    ros::MultiThreadedSpinner spinner(2);
    spinner.spin();

    return 0;
}