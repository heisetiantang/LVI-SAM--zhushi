#include "utility.h"
#include "lvi_sam/cloud_info.h"

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>

#include <gtsam/nonlinear/ISAM2.h>

using namespace gtsam;

using symbol_shorthand::B;  // Bias  (ax,ay,az,gx,gy,gz)
using symbol_shorthand::G;  // GPS pose
using symbol_shorthand::V;  // Vel   (xdot,ydot,zdot)
using symbol_shorthand::X;  // Pose3 (x,y,z,r,p,y)

/*
 * A point cloud type that has 6D pose info ([x,y,z,roll,pitch,yaw] intensity is time stamp)
 */
struct PointXYZIRPYT
{
  PCL_ADD_POINT4D
  PCL_ADD_INTENSITY;
  float roll;
  float pitch;
  float yaw;
  double time;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIRPYT, (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(float, roll, roll)(float, pitch, pitch)(float, yaw, yaw)(double, time, time))

typedef PointXYZIRPYT PointTypePose;

class mapOptimization : public ParamServer
{
 public:
  // gtsam
  NonlinearFactorGraph gtSAMgraph;
  Values initialEstimate;
  Values optimizedEstimate;
  ISAM2 *isam;
  Values isamCurrentEstimate;
  Eigen::MatrixXd poseCovariance;

  ros::Publisher pubLaserCloudSurround;
  ros::Publisher pubOdomAftMappedROS;
  ros::Publisher pubKeyPoses;
  ros::Publisher pubPath;

  ros::Publisher pubHistoryKeyFrames;
  ros::Publisher pubIcpKeyFrames;
  ros::Publisher pubRecentKeyFrames;
  ros::Publisher pubRecentKeyFrame;
  ros::Publisher pubCloudRegisteredRaw;
  ros::Publisher pubLoopConstraintEdge;

  ros::Subscriber subLaserCloudInfo;
  ros::Subscriber subGPS;
  ros::Subscriber subLoopInfo;

  std::deque<nav_msgs::Odometry> gpsQueue;
  lvi_sam::cloud_info cloudInfo;

  vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyFrames;
  vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;

  pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;
  pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;

  pcl::PointCloud<PointType>::Ptr laserCloudCornerLast;    // corner feature set from odoOptimization  odoOptimization 的角点特征集
  pcl::PointCloud<PointType>::Ptr laserCloudSurfLast;      // surf feature set from odoOptimization  odoOptimization 的平面特征集
  pcl::PointCloud<PointType>::Ptr laserCloudCornerLastDS;  // downsampled corner featuer set from odoOptimization   优化后的角点特征集
  pcl::PointCloud<PointType>::Ptr laserCloudSurfLastDS;    // downsampled surf featuer set from odoOptimization    优化后的平面特征集

  pcl::PointCloud<PointType>::Ptr laserCloudOri;
  pcl::PointCloud<PointType>::Ptr coeffSel;

  std::vector<PointType> laserCloudOriCornerVec;  // corner point holder for parallel computation 用于并行计算的角点支架
  std::vector<PointType> coeffSelCornerVec;
  std::vector<bool> laserCloudOriCornerFlag;
  std::vector<PointType> laserCloudOriSurfVec;  // surf point holder for parallel computation  用于并行计算的面点点支架
  std::vector<PointType> coeffSelSurfVec;
  std::vector<bool> laserCloudOriSurfFlag;

  pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap;
  pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap;
  pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMapDS;
  pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS;

  pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap;
  pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap;

  pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses;
  pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses;

  pcl::PointCloud<PointType>::Ptr latestKeyFrameCloud;
  pcl::PointCloud<PointType>::Ptr nearHistoryKeyFrameCloud;









  pcl::VoxelGrid<PointType> downSizeFilterCorner;
  pcl::VoxelGrid<PointType> downSizeFilterSurf;
  pcl::VoxelGrid<PointType> downSizeFilterICP;
  pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyPoses;  // for surrounding key poses of scan-to-map optimization 用于扫描到地图优化的周围关键姿势












  ros::Time timeLaserInfoStamp;
  double timeLaserInfoCur;

  float transformTobeMapped[6];

  std::mutex mtx;

  bool isDegenerate = false;
  cv::Mat matP;

  int laserCloudCornerLastDSNum = 0;
  int laserCloudSurfLastDSNum = 0;

  bool aLoopIsClosed = false;
  int imuPreintegrationResetId = 0;

  nav_msgs::Path globalPath;

  Eigen::Affine3f transPointAssociateToMap;

  map<int, int> loopIndexContainer;  // from new to old
  vector<pair<int, int>> loopIndexQueue;
  vector<gtsam::Pose3> loopPoseQueue;
  vector<gtsam::noiseModel::Diagonal::shared_ptr> loopNoiseQueue;

  mapOptimization()
  {
    //定义ISAM2参数类对象Parameters
    ISAM2Params parameters;
    //重线性化阈值，求雅阁比用
    parameters.relinearizeThreshold = 0.1;
    ///< Only relinearize any variables every
        ///< relinearizeSkip calls to ISAM2::update (default:
        ///< 10)
    //每调用1次就线性化一次
    parameters.relinearizeSkip = 1;
      /* The typical cycle of using this class to create an instance :
           1.【by providing ISAM2Params to the constructor】, 即isam = new ISAM2(parameters);
           2.then add measurements and variables as they arrive using the update() method.
           3.At any time, calculateEstimate() may be
         * called to obtain the current estimate of all variables.
         */
    isam = new ISAM2(parameters);//ISAM2用来优化
    //创建发布
    pubKeyPoses = nh.advertise<sensor_msgs::PointCloud2>(PROJECT_NAME + "/lidar/mapping/trajectory", 1);
    pubLaserCloudSurround = nh.advertise<sensor_msgs::PointCloud2>(PROJECT_NAME + "/lidar/mapping/map_global", 1);
    pubOdomAftMappedROS = nh.advertise<nav_msgs::Odometry>(PROJECT_NAME + "/lidar/mapping/odometry", 1);
    pubPath = nh.advertise<nav_msgs::Path>(PROJECT_NAME + "/lidar/mapping/path", 1);



//***********************************
  //接收点云消息
  //接收featureExtraction节点laserCloundInfoHandle函数的点云特征消息，在image中进行特征提取，面特征和线特征，这里接受
    subLaserCloudInfo = nh.subscribe<lvi_sam::cloud_info>(PROJECT_NAME + "/lidar/feature/cloud_info", 5, &mapOptimization::laserCloudInfoHandler, this, ros::TransportHints().tcpNoDelay());
  //接收GPS信息，这里是没有使用的
    subGPS = nh.subscribe<nav_msgs::Odometry>(gpsTopic, 50, &mapOptimization::gpsHandler, this, ros::TransportHints().tcpNoDelay());
    //回环消息
    subLoopInfo = nh.subscribe<std_msgs::Float64MultiArray>(PROJECT_NAME + "/vins/loop/match_frame", 5, &mapOptimization::loopHandler, this, ros::TransportHints().tcpNoDelay());
//**********************************



    //创建发布
    //历史关键帧
    pubHistoryKeyFrames = nh.advertise<sensor_msgs::PointCloud2>(PROJECT_NAME + "/lidar/mapping/loop_closure_history_cloud", 1);
    //ICP关键帧
    pubIcpKeyFrames = nh.advertise<sensor_msgs::PointCloud2>(PROJECT_NAME + "/lidar/mapping/loop_closure_corrected_cloud", 1);
    //回环约束可视化信息
    pubLoopConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>(PROJECT_NAME + "/lidar/mapping/loop_closure_constraints", 1);
    //局部地图（只有面特征，因为数量多）
    pubRecentKeyFrames = nh.advertise<sensor_msgs::PointCloud2>(PROJECT_NAME + "/lidar/mapping/map_local", 1);
    //当前帧点云
    pubRecentKeyFrame = nh.advertise<sensor_msgs::PointCloud2>(PROJECT_NAME + "/lidar/mapping/cloud_registered", 1);
    pubCloudRegisteredRaw = nh.advertise<sensor_msgs::PointCloud2>(PROJECT_NAME + "/lidar/mapping/cloud_registered_raw", 1);

    //点云降采样，发布地图的滤波器
    //角点云 0.2
    downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
    //面点云 0.4
    downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
    downSizeFilterICP.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
    //临近关键帧 2.0 。将关键帧按照点云的方式进行存储，方便使用kd树寻找
    downSizeFilterSurroundingKeyPoses.setLeafSize(surroundingKeyframeDensity, surroundingKeyframeDensity, surroundingKeyframeDensity);  // for surrounding key poses of scan-to-map optimization

    allocateMemory();//初始化
  }

  void allocateMemory()
  {
    cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());//点云指针。关键帧位置的集合
    cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());//关键帧位姿的集合

    kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());//kd树构造
    kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());

    // 来自 odo 优化的角点特征集
    // 来自 odo 优化的 surf 特征集
    // 来自 odoOptimization 的下采样角点特征集
    // 来自 odoOptimization 的下采样 surf 特征集
    laserCloudCornerLast.reset(new pcl::PointCloud<PointType>());    // corner feature set from odoOptimization当前帧角点云
    laserCloudSurfLast.reset(new pcl::PointCloud<PointType>());      // surf feature set from odoOptimization当前帧面点云
    laserCloudCornerLastDS.reset(new pcl::PointCloud<PointType>());  // downsampled corner featuer set from odoOptimization
    laserCloudSurfLastDS.reset(new pcl::PointCloud<PointType>());    // downsampled surf featuer set from odoOptimization

    laserCloudOri.reset(new pcl::PointCloud<PointType>());//激光点云
    coeffSel.reset(new pcl::PointCloud<PointType>());//存储残差

    laserCloudOriCornerVec.resize(N_SCAN * Horizon_SCAN);
    coeffSelCornerVec.resize(N_SCAN * Horizon_SCAN);
    laserCloudOriCornerFlag.resize(N_SCAN * Horizon_SCAN);
    laserCloudOriSurfVec.resize(N_SCAN * Horizon_SCAN);
    coeffSelSurfVec.resize(N_SCAN * Horizon_SCAN);
    laserCloudOriSurfFlag.resize(N_SCAN * Horizon_SCAN);

    std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
    std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);

    laserCloudCornerFromMap.reset(new pcl::PointCloud<PointType>());
    laserCloudSurfFromMap.reset(new pcl::PointCloud<PointType>());
    laserCloudCornerFromMapDS.reset(new pcl::PointCloud<PointType>());
    laserCloudSurfFromMapDS.reset(new pcl::PointCloud<PointType>());

    kdtreeCornerFromMap.reset(new pcl::KdTreeFLANN<PointType>());
    kdtreeSurfFromMap.reset(new pcl::KdTreeFLANN<PointType>());

    latestKeyFrameCloud.reset(new pcl::PointCloud<PointType>());
    nearHistoryKeyFrameCloud.reset(new pcl::PointCloud<PointType>());

    for (int i = 0; i < 6; ++i) { transformTobeMapped[i] = 0; }//激光雷达坐标系到世界坐标系的变换

    matP = cv::Mat(6, 6, CV_32F, cv::Scalar::all(0));//退化矩阵
  }

  //1.
  //点云话题回调函数
  void laserCloudInfoHandler(const lvi_sam::cloud_infoConstPtr &msgIn)//自定义点云消息
  {

    timeLaserInfoStamp = msgIn->header.stamp;    // extract time stamp // 提取时间戳
    timeLaserInfoCur = msgIn->header.stamp.toSec(); //时间戳转换成秒

    // extract info ana feature cloud// 边缘点和平面点点云
    cloudInfo = *msgIn;//点云信息指针指向点云信息消息
    pcl::fromROSMsg(msgIn->cloud_corner, *laserCloudCornerLast);//格式转换。ros信息转换为pcl格式。角点和面点各用一个指针指着
    pcl::fromROSMsg(msgIn->cloud_surface, *laserCloudSurfLast);

    std::lock_guard<std::mutex> lock(mtx);
    //考虑到优化的速度可能小于激光雷达的0.1秒，因此来两帧才处理一下
    static double timeLastProcessing = -1;
    //点云当前的时间-上次处理的时间 > (地图处理的时间间隔0.15)
    if (timeLaserInfoCur - timeLastProcessing >= mappingProcessInterval)//调频
    {
      //上次处理时间改为当前时间 用于下一次处理时对比时间差
      timeLastProcessing = timeLaserInfoCur;
      //1.1  在lvisam中    使用VINS和九轴的imu磁力计提供初始姿态变化
      // 计算点云的先验位姿 (通过imu或者vins odom)//更新初始位姿估计
      updateInitialGuess();

      //1.2 //提取附近关键帧
      extractSurroundingKeyFrames();
      //1.3//降采样当前扫描
      downsampleCurrentScan();
      //1.4 
      //扫描到地图的匹配与优化  这里与LEGO差不多思想
      //根据现有地图与最新点云数据进行配准从而更新机器人精确位姿与融合建图，
      scan2MapOptimization();
      //1.5//保存关键帧 这里大致与Lego流程一致 但是与LOAM就有区别了
      saveKeyFramesAndFactor();
      //1.6//更新位姿
      correctPoses();
      //1.7//发布里程计
      publishOdometry();
      //1.8//发布关键帧点云
      publishFrames();
    }


  }

  //2.
  void gpsHandler(const nav_msgs::Odometry::ConstPtr &gpsMsg)
  {
    std::lock_guard<std::mutex> lock(mtx);
    gpsQueue.push_back(*gpsMsg);
  }

//******************************************************************************************
//a工具性质函数：
//a1第i帧的点转换到第一帧坐标系下
  void pointAssociateToMap(PointType const *const pi, PointType *const po)
  {
    po->x = transPointAssociateToMap(0, 0) * pi->x + transPointAssociateToMap(0, 1) * pi->y + transPointAssociateToMap(0, 2) * pi->z + transPointAssociateToMap(0, 3);
    po->y = transPointAssociateToMap(1, 0) * pi->x + transPointAssociateToMap(1, 1) * pi->y + transPointAssociateToMap(1, 2) * pi->z + transPointAssociateToMap(1, 3);
    po->z = transPointAssociateToMap(2, 0) * pi->x + transPointAssociateToMap(2, 1) * pi->y + transPointAssociateToMap(2, 2) * pi->z + transPointAssociateToMap(2, 3);
    po->intensity = pi->intensity;
  }

//a2点云输入和转换输入
  pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose *transformIn)
  {
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

    PointType *pointFrom;

    int cloudSize = cloudIn->size();
    cloudOut->resize(cloudSize);
    //获得变换矩阵
    Eigen::Affine3f transCur = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);

    for (int i = 0; i < cloudSize; ++i)
    {
      pointFrom = &cloudIn->points[i];
      cloudOut->points[i].x = transCur(0, 0) * pointFrom->x + transCur(0, 1) * pointFrom->y + transCur(0, 2) * pointFrom->z + transCur(0, 3);
      cloudOut->points[i].y = transCur(1, 0) * pointFrom->x + transCur(1, 1) * pointFrom->y + transCur(1, 2) * pointFrom->z + transCur(1, 3);
      cloudOut->points[i].z = transCur(2, 0) * pointFrom->x + transCur(2, 1) * pointFrom->y + transCur(2, 2) * pointFrom->z + transCur(2, 3);
      cloudOut->points[i].intensity = pointFrom->intensity;
    }
    return cloudOut;
  }

  //返回用于gtsam的pose
  gtsam::Pose3 affine3fTogtsamPose3(const Eigen::Affine3f &thisPose)
  {
    float x, y, z, roll, pitch, yaw;
    pcl::getTranslationAndEulerAngles(thisPose, x, y, z, roll, pitch, yaw);
    //由变换矩阵返回用于gtsam的pose
    return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(roll), double(pitch), double(yaw)), gtsam::Point3(double(x), double(y), double(z)));
  }

  gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint) { return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)), gtsam::Point3(double(thisPoint.x), double(thisPoint.y), double(thisPoint.z))); }

  gtsam::Pose3 trans2gtsamPose(float transformIn[]) { return gtsam::Pose3(gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]), gtsam::Point3(transformIn[3], transformIn[4], transformIn[5])); }

  Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint) { return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw); }

  //由变换矩阵返回仿射变换矩阵
  Eigen::Affine3f trans2Affine3f(float transformIn[]) { return pcl::getTransformation(transformIn[3], transformIn[4], transformIn[5], transformIn[0], transformIn[1], transformIn[2]); }

  PointTypePose trans2PointTypePose(float transformIn[])
  {
    PointTypePose thisPose6D;
    thisPose6D.x = transformIn[3];
    thisPose6D.y = transformIn[4];
    thisPose6D.z = transformIn[5];
    thisPose6D.roll = transformIn[0];
    thisPose6D.pitch = transformIn[1];
    thisPose6D.yaw = transformIn[2];
    return thisPose6D;
  }

  void visualizeGlobalMapThread()
  {
    ros::Rate rate(0.2);
    while (ros::ok())
    {
      rate.sleep();
      publishGlobalMap();
    }

    if (savePCD == false)
      return;

    cout << "****************************************************" << endl;
    cout << "Saving map to pcd files ..." << endl;
    // create directory and remove old files;
    savePCDDirectory = std::getenv("HOME") + savePCDDirectory;
    int unused = system((std::string("exec rm -r ") + savePCDDirectory).c_str());
    unused = system((std::string("mkdir ") + savePCDDirectory).c_str());
    ++unused;
    // save key frame transformations
    pcl::io::savePCDFileASCII(savePCDDirectory + "trajectory.pcd", *cloudKeyPoses3D);
    pcl::io::savePCDFileASCII(savePCDDirectory + "transformations.pcd", *cloudKeyPoses6D);
    // extract global point cloud map
    pcl::PointCloud<PointType>::Ptr globalCornerCloud(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalSurfCloud(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalMapCloud(new pcl::PointCloud<PointType>());
    for (int i = 0; i < (int) cloudKeyPoses3D->size(); i++)
    {
      // clip cloud
      // pcl::PointCloud<PointType>::Ptr cornerTemp(new pcl::PointCloud<PointType>());
      // pcl::PointCloud<PointType>::Ptr cornerTemp2(new pcl::PointCloud<PointType>());
      // *cornerTemp = *transformPointCloud(cornerCloudKeyFrames[i],  &cloudKeyPoses6D->points[i]);
      // for (int j = 0; j < (int)cornerTemp->size(); ++j)
      // {
      //     if (cornerTemp->points[j].z > cloudKeyPoses6D->points[i].z && cornerTemp->points[j].z < cloudKeyPoses6D->points[i].z + 5)
      //         cornerTemp2->push_back(cornerTemp->points[j]);
      // }
      // pcl::PointCloud<PointType>::Ptr surfTemp(new pcl::PointCloud<PointType>());
      // pcl::PointCloud<PointType>::Ptr surfTemp2(new pcl::PointCloud<PointType>());
      // *surfTemp = *transformPointCloud(surfCloudKeyFrames[i],  &cloudKeyPoses6D->points[i]);
      // for (int j = 0; j < (int)surfTemp->size(); ++j)
      // {
      //     if (surfTemp->points[j].z > cloudKeyPoses6D->points[i].z && surfTemp->points[j].z < cloudKeyPoses6D->points[i].z + 5)
      //         surfTemp2->push_back(surfTemp->points[j]);
      // }
      // *globalCornerCloud += *cornerTemp2;
      // *globalSurfCloud   += *surfTemp2;

      // origin cloud
      *globalCornerCloud += *transformPointCloud(cornerCloudKeyFrames[i], &cloudKeyPoses6D->points[i]);
      *globalSurfCloud += *transformPointCloud(surfCloudKeyFrames[i], &cloudKeyPoses6D->points[i]);
      cout << "\r" << std::flush << "Processing feature cloud " << i << " of " << cloudKeyPoses6D->size() << " ...";
    }
    // down-sample and save corner cloud
    downSizeFilterCorner.setInputCloud(globalCornerCloud);
    pcl::io::savePCDFileASCII(savePCDDirectory + "cloudCorner.pcd", *globalCornerCloud);
    // down-sample and save surf cloud
    downSizeFilterSurf.setInputCloud(globalSurfCloud);
    pcl::io::savePCDFileASCII(savePCDDirectory + "cloudSurf.pcd", *globalSurfCloud);
    // down-sample and save global point cloud map
    *globalMapCloud += *globalCornerCloud;
    *globalMapCloud += *globalSurfCloud;
    downSizeFilterSurf.setInputCloud(globalMapCloud);
    pcl::io::savePCDFileASCII(savePCDDirectory + "cloudGlobal.pcd", *globalMapCloud);
    cout << "****************************************************" << endl;
    cout << "Saving map to pcd files completed" << endl;
  }

  void publishGlobalMap()
  {
    if (pubLaserCloudSurround.getNumSubscribers() == 0)
      return;

    if (cloudKeyPoses3D->points.empty() == true)
      return;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMap(new pcl::KdTreeFLANN<PointType>());
    ;
    pcl::PointCloud<PointType>::Ptr globalMapKeyPoses(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalMapKeyPosesDS(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalMapKeyFrames(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalMapKeyFramesDS(new pcl::PointCloud<PointType>());

    // kd-tree to find near key frames to visualize
    std::vector<int> pointSearchIndGlobalMap;
    std::vector<float> pointSearchSqDisGlobalMap;
    // search near key frames to visualize
    mtx.lock();
    kdtreeGlobalMap->setInputCloud(cloudKeyPoses3D);
    kdtreeGlobalMap->radiusSearch(cloudKeyPoses3D->back(), globalMapVisualizationSearchRadius, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);
    mtx.unlock();

    for (int i = 0; i < (int) pointSearchIndGlobalMap.size(); ++i) globalMapKeyPoses->push_back(cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]);
    // downsample near selected key frames
    pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyPoses;                                                                                             // for global map visualization
    downSizeFilterGlobalMapKeyPoses.setLeafSize(globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity);  // for global map visualization
    downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
    downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);

    // extract visualized and downsampled key frames
    for (int i = 0; i < (int) globalMapKeyPosesDS->size(); ++i)
    {
      if (pointDistance(globalMapKeyPosesDS->points[i], cloudKeyPoses3D->back()) > globalMapVisualizationSearchRadius)
        continue;
      int thisKeyInd = (int) globalMapKeyPosesDS->points[i].intensity;
      *globalMapKeyFrames += *transformPointCloud(cornerCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
      *globalMapKeyFrames += *transformPointCloud(surfCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
    }
    // downsample visualized points
    pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames;                                                                                    // for global map visualization
    downSizeFilterGlobalMapKeyFrames.setLeafSize(globalMapVisualizationLeafSize, globalMapVisualizationLeafSize, globalMapVisualizationLeafSize);  // for global map visualization
    downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
    downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS);
    publishCloud(&pubLaserCloudSurround, globalMapKeyFramesDS, timeLaserInfoStamp, "odom");
  }

  // 线程: 通过vins进行闭环检测
  void loopHandler(const std_msgs::Float64MultiArray::ConstPtr &loopMsg)
  {
    // control loop closure frequency
    static double last_loop_closure_time = -1;
    {
      // std::lock_guard<std::mutex> lock(mtx);
      if (timeLaserInfoCur - last_loop_closure_time < 5.0)
        return;
      else
        last_loop_closure_time = timeLaserInfoCur;
    }

    performLoopClosure(*loopMsg);
  }

  // 闭环检测 (通过 距离 或者 vins 得到的闭环候选帧), loopMsg保存的是时间戳(当前帧, 闭环帧);
  void performLoopClosure(const std_msgs::Float64MultiArray &loopMsg)
  {
    // 获取所有关键帧的位姿
    pcl::PointCloud<PointTypePose>::Ptr copy_cloudKeyPoses6D(new pcl::PointCloud<PointTypePose>());
    {
      std::lock_guard<std::mutex> lock(mtx);
      *copy_cloudKeyPoses6D = *cloudKeyPoses6D;
    }

    // 1.通过loopMsg的时间戳来寻找 闭环候选帧
    // get lidar keyframe id
    int key_cur = -1;  // latest lidar keyframe id
    int key_pre = -1;  // previous lidar keyframe id
    {
      // 通过loopMsg的时间戳来寻找 闭环候选帧
      loopFindKey(loopMsg, copy_cloudKeyPoses6D, key_cur, key_pre);
      if (key_cur == -1 || key_pre == -1 || key_cur == key_pre)  // || abs(key_cur - key_pre) < 25)
        return;
    }

    // check if loop added before
    {
      // if image loop closure comes at high frequency, many image loop may point to the same key_cur
      auto it = loopIndexContainer.find(key_cur);
      if (it != loopIndexContainer.end())
        return;
    }

    // 3.分别为当前帧和闭环帧构造localmap, 进行map to map的闭环匹配
    // get lidar keyframe cloud
    pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>());
    {
      loopFindNearKeyframes(copy_cloudKeyPoses6D, cureKeyframeCloud, key_cur, 0);
      loopFindNearKeyframes(copy_cloudKeyPoses6D, prevKeyframeCloud, key_pre, historyKeyframeSearchNum);
      if (cureKeyframeCloud->size() < 300 || prevKeyframeCloud->size() < 1000)
        return;
      // 发布 闭环帧的localmap点云
      if (pubHistoryKeyFrames.getNumSubscribers() != 0)
        publishCloud(&pubHistoryKeyFrames, prevKeyframeCloud, timeLaserInfoStamp, "odom");
    }

    // get keyframe pose
    Eigen::Affine3f pose_cur;     // 当前帧pose
    Eigen::Affine3f pose_pre;     // 闭环帧pose
    Eigen::Affine3f pose_diff_t;  // serves as initial guess 将两者的相对位姿作为初始位姿
    {
      pose_cur = pclPointToAffine3f(copy_cloudKeyPoses6D->points[key_cur]);
      pose_pre = pclPointToAffine3f(copy_cloudKeyPoses6D->points[key_pre]);

      Eigen::Vector3f t_diff;
      t_diff.x() = -(pose_cur.translation().x() - pose_pre.translation().x());
      t_diff.y() = -(pose_cur.translation().y() - pose_pre.translation().y());
      t_diff.z() = -(pose_cur.translation().z() - pose_pre.translation().z());
      if (t_diff.norm() < historyKeyframeSearchRadius)
        t_diff.setZero();
      pose_diff_t = pcl::getTransformation(t_diff.x(), t_diff.y(), t_diff.z(), 0, 0, 0);
    }

    // 5.使用icp进行闭环匹配(map to map)

    // transform and rotate cloud for matching
    pcl::IterativeClosestPoint<PointType, PointType> icp;
    // pcl::GeneralizedIterativeClosestPoint<PointType, PointType> icp;
    icp.setMaxCorrespondenceDistance(historyKeyframeSearchRadius * 2);
    icp.setMaximumIterations(100);
    icp.setRANSACIterations(0);
    icp.setTransformationEpsilon(1e-3);
    icp.setEuclideanFitnessEpsilon(1e-3);

    // initial guess cloud
    // 根据初始相对位姿, 对当前帧点云进行transform
    pcl::PointCloud<PointType>::Ptr cureKeyframeCloud_new(new pcl::PointCloud<PointType>());
    pcl::transformPointCloud(*cureKeyframeCloud, *cureKeyframeCloud_new, pose_diff_t);

    // match using icp
    icp.setInputSource(cureKeyframeCloud_new);
    icp.setInputTarget(prevKeyframeCloud);
    pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
    icp.align(*unused_result);

    if (pubIcpKeyFrames.getNumSubscribers() != 0)
    {
      pcl::PointCloud<PointType>::Ptr closed_cloud(new pcl::PointCloud<PointType>());
      pcl::transformPointCloud(*cureKeyframeCloud_new, *closed_cloud, icp.getFinalTransformation());
      publishCloud(&pubIcpKeyFrames, closed_cloud, timeLaserInfoStamp, "odom");
    }

    // add graph factor
    // 将闭环保存至loopIndexQueue loopPoseQueue loopNoiseQueue中供addLoopFactor()使用
    if (icp.getFitnessScore() < historyKeyframeFitnessScore && icp.hasConverged() == true)
    {
      // get gtsam pose
      gtsam::Pose3 poseFrom = affine3fTogtsamPose3(Eigen::Affine3f(icp.getFinalTransformation()) * pose_diff_t * pose_cur);
      gtsam::Pose3 poseTo = pclPointTogtsamPose3(copy_cloudKeyPoses6D->points[key_pre]);
      // get noise
      float noise = icp.getFitnessScore();
      gtsam::Vector Vector6(6);
      Vector6 << noise, noise, noise, noise, noise, noise;
      noiseModel::Diagonal::shared_ptr constraintNoise = noiseModel::Diagonal::Variances(Vector6);
      // save pose constraint
      mtx.lock();
      loopIndexQueue.push_back(make_pair(key_cur, key_pre));
      loopPoseQueue.push_back(poseFrom.between(poseTo));
      loopNoiseQueue.push_back(constraintNoise);
      mtx.unlock();
      // add loop pair to container
      loopIndexContainer[key_cur] = key_pre;
    }

    // visualize loop constraints 发布 所有闭环约束
    if (!loopIndexContainer.empty())
    {
      visualization_msgs::MarkerArray markerArray;
      // loop nodes
      visualization_msgs::Marker markerNode;
      markerNode.header.frame_id = "odom";
      markerNode.header.stamp = timeLaserInfoStamp;
      markerNode.action = visualization_msgs::Marker::ADD;
      markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
      markerNode.ns = "loop_nodes";
      markerNode.id = 0;
      markerNode.pose.orientation.w = 1;
      markerNode.scale.x = 0.3;
      markerNode.scale.y = 0.3;
      markerNode.scale.z = 0.3;
      markerNode.color.r = 0;
      markerNode.color.g = 0.8;
      markerNode.color.b = 1;
      markerNode.color.a = 1;
      // loop edges
      visualization_msgs::Marker markerEdge;
      markerEdge.header.frame_id = "odom";
      markerEdge.header.stamp = timeLaserInfoStamp;
      markerEdge.action = visualization_msgs::Marker::ADD;
      markerEdge.type = visualization_msgs::Marker::LINE_LIST;
      markerEdge.ns = "loop_edges";
      markerEdge.id = 1;
      markerEdge.pose.orientation.w = 1;
      markerEdge.scale.x = 0.1;
      markerEdge.color.r = 0.9;
      markerEdge.color.g = 0.9;
      markerEdge.color.b = 0;
      markerEdge.color.a = 1;

      for (auto it = loopIndexContainer.begin(); it != loopIndexContainer.end(); ++it)
      {
        int key_cur = it->first;
        int key_pre = it->second;
        geometry_msgs::Point p;
        p.x = copy_cloudKeyPoses6D->points[key_cur].x;
        p.y = copy_cloudKeyPoses6D->points[key_cur].y;
        p.z = copy_cloudKeyPoses6D->points[key_cur].z;
        markerNode.points.push_back(p);
        markerEdge.points.push_back(p);
        p.x = copy_cloudKeyPoses6D->points[key_pre].x;
        p.y = copy_cloudKeyPoses6D->points[key_pre].y;
        p.z = copy_cloudKeyPoses6D->points[key_pre].z;
        markerNode.points.push_back(p);
        markerEdge.points.push_back(p);
      }

      markerArray.markers.push_back(markerNode);
      markerArray.markers.push_back(markerEdge);
      pubLoopConstraintEdge.publish(markerArray);
    }
  }

  // 为关键帧key构造localmap点云(nearKeyframes)
  void loopFindNearKeyframes(const pcl::PointCloud<PointTypePose>::Ptr &copy_cloudKeyPoses6D, pcl::PointCloud<PointType>::Ptr &nearKeyframes, const int &key, const int &searchNum)
  {
    // extract near keyframes
    nearKeyframes->clear();
    int cloudSize = copy_cloudKeyPoses6D->size();
    for (int i = -searchNum; i <= searchNum; ++i)
    {
      int key_near = key + i;
      if (key_near < 0 || key_near >= cloudSize)
        continue;
      *nearKeyframes += *transformPointCloud(cornerCloudKeyFrames[key_near], &copy_cloudKeyPoses6D->points[key_near]);
      *nearKeyframes += *transformPointCloud(surfCloudKeyFrames[key_near], &copy_cloudKeyPoses6D->points[key_near]);
    }

    if (nearKeyframes->empty())
      return;

    // downsample near keyframes
    pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
    downSizeFilterICP.setInputCloud(nearKeyframes);
    downSizeFilterICP.filter(*cloud_temp);
    *nearKeyframes = *cloud_temp;
  }

  // 通过loopMsg的时间戳来寻找 闭环候选帧
  void loopFindKey(const std_msgs::Float64MultiArray &loopMsg, const pcl::PointCloud<PointTypePose>::Ptr &copy_cloudKeyPoses6D, int &key_cur, int &key_pre)
  {
    if (loopMsg.data.size() != 2)
      return;

    double loop_time_cur = loopMsg.data[0];
    double loop_time_pre = loopMsg.data[1];

    // 时间戳在25s之内, 不是闭环
    if (abs(loop_time_cur - loop_time_pre) < historyKeyframeSearchTimeDiff)
      return;

    int cloudSize = copy_cloudKeyPoses6D->size();
    if (cloudSize < 2)
      return;

    // latest key
    key_cur = cloudSize - 1;  // 当前帧
    for (int i = cloudSize - 1; i >= 0; --i)
    {
      if (copy_cloudKeyPoses6D->points[i].time > loop_time_cur)
        key_cur = round(copy_cloudKeyPoses6D->points[i].intensity);
      else
        break;
    }

    // previous key
    key_pre = 0;  // 闭环帧
    for (int i = 0; i < cloudSize; ++i)
    {
      if (copy_cloudKeyPoses6D->points[i].time < loop_time_pre)
        key_pre = round(copy_cloudKeyPoses6D->points[i].intensity);
      else
        break;
    }
  }

  // 线程: 通过距离进行闭环检测
  void loopClosureThread()
  {
    if (loopClosureEnableFlag == false)
      return;

    ros::Rate rate(0.5);  // 每2s进行一次闭环检测
    while (ros::ok())
    {
      rate.sleep();
      performLoopClosureDetection();
    }
  }

  // 通过距离进行闭环检测
  void performLoopClosureDetection()
  {
    std::vector<int> pointSearchIndLoop;
    std::vector<float> pointSearchSqDisLoop;

    // 通过距离找到的闭环候选帧
    int key_cur = -1;
    int key_pre = -1;

    double loop_time_cur = -1;
    double loop_time_pre = -1;

    // find latest key and time
    // 1.使用kdtree寻找最近的keyframes, 作为闭环检测的候选关键帧 (20m以内)
    {
      std::lock_guard<std::mutex> lock(mtx);

      if (cloudKeyPoses3D->empty())
        return;

      kdtreeHistoryKeyPoses->setInputCloud(cloudKeyPoses3D);
      kdtreeHistoryKeyPoses->radiusSearch(cloudKeyPoses3D->back(), historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 0);

      key_cur = cloudKeyPoses3D->size() - 1;
      loop_time_cur = cloudKeyPoses6D->points[key_cur].time;
    }

    // find previous key and time
    // 2.在候选关键帧集合中，找到与当前帧时间相隔较远的最近帧，设为候选匹配帧 (25s之前)
    {
      for (int i = 0; i < (int) pointSearchIndLoop.size(); ++i)
      {
        int id = pointSearchIndLoop[i];
        if (abs(cloudKeyPoses6D->points[id].time - loop_time_cur) > historyKeyframeSearchTimeDiff)
        {
          key_pre = id;
          loop_time_pre = cloudKeyPoses6D->points[key_pre].time;
          break;
        }
      }
    }

    // 未检测到闭环
    if (key_cur == -1 || key_pre == -1 || key_pre == key_cur || loop_time_cur < 0 || loop_time_pre < 0)
      return;

    std_msgs::Float64MultiArray match_msg;
    match_msg.data.push_back(loop_time_cur);  // 当前帧时间戳
    match_msg.data.push_back(loop_time_pre);  // 闭环帧时间戳
    performLoopClosure(match_msg);
  }

  //1.1计算点云的先验位姿 (通过imu或者vins odom)
  void updateInitialGuess()
  { 
    //保存当前的变换 第一次全是0
    incrementalOdometryAffineFront = trans2Affine3f(transformTobeMapped);
    // initialization 初始化第一次用imageProjection中获得的imu的欧拉角初始化变换参数
   
    // 第一帧点云, 直接使用imu磁力计进行初始化，而不是使用VINS来进行
    static Eigen::Affine3f lastImuTransformation;
    // system initialization
    if (cloudKeyPoses3D->points.empty())
    {
      //初始位姿由IMU磁力计得到
      transformTobeMapped[0] = cloudInfo.imuRollInit;
      transformTobeMapped[1] = cloudInfo.imuPitchInit;
      transformTobeMapped[2] = cloudInfo.imuYawInit;
      //在params.yaml中说明如果用了gps数据则useImuHeadingInitialization是true
      if (!useImuHeadingInitialization)
        transformTobeMapped[2] = 0;

      // 保存下来, 给下一帧使用。将imu提供的欧拉角和平移置0转换为eigen格式的静态变量lastImuTransformation（4*4矩阵）
      lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit);  // save imu before return;
      return;//直接出函数
    }

    //非第一帧
    //使用 VINS 里程计估计进行姿态猜测
    //使用IMU预积分的估计来进行位姿估计 上一段初始化只执行一次 下面的不是 满足条件后会多次执行
    static int odomResetId = 0;
    static bool lastVinsTransAvailable = false;
    static Eigen::Affine3f lastVinsTransformation;  // 上一帧的vins odom
    //如果发来的VINS里程计信息可用
    if (cloudInfo.odomAvailable == true && cloudInfo.odomResetId == odomResetId)
    {
      
      // ROS_INFO("Using VINS initial guess");//lastVinsTransAvailable把关
      // 如果里程计可用,去检查是否【第一次】收到VINS提供的里程计消息 分成了两个if else
            // ROS_INFO("Using VINS initial guess");
            // lastVinsTransAvailable == false表示【第一次】收到VINS提供的里程计消息
      if (lastVinsTransAvailable == false)
      {
        // vins重新启动了, 保存vins重启后的第一帧odom
        // ROS_INFO("Initializing VINS initial guess");
         // 如果【第一次】收到VINS提供的里程计消息
                // 将VINS IMU递推的里程计(xyz+欧拉角)转换为Eigen::Affine3f的lastVinsTransformation
                // 注意寻找cloudInfo.odom***数据来源的方式
        lastVinsTransformation = pcl::getTransformation(cloudInfo.odomX, cloudInfo.odomY, cloudInfo.odomZ, cloudInfo.odomRoll, cloudInfo.odomPitch, cloudInfo.odomYaw);
        lastVinsTransAvailable = true;
      }
      else
      {
        // 2.通过vins odom计算点云的先验位姿
        // ROS_INFO("Obtaining VINS incremental guess");
        // 程序执行到这里，说明已经不是【第一次】收到VINS提供的里程计消息
                // 则将【当前】VINS IMU递推的里程计转换为Eigen::Affine3f的transBack
        Eigen::Affine3f transBack = pcl::getTransformation(cloudInfo.odomX, cloudInfo.odomY, cloudInfo.odomZ, cloudInfo.odomRoll, cloudInfo.odomPitch, cloudInfo.odomYaw);
        Eigen::Affine3f transIncre = lastVinsTransformation.inverse() * transBack;

        Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);
        Eigen::Affine3f transFinal = transTobe * transIncre;
        pcl::getTranslationAndEulerAngles(transFinal, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);

        // 保存下来, 为下一帧使用
        lastVinsTransformation = pcl::getTransformation(cloudInfo.odomX, cloudInfo.odomY, cloudInfo.odomZ, cloudInfo.odomRoll, cloudInfo.odomPitch, cloudInfo.odomYaw);

        lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit);  // save imu before return;
        return;
      }
    }
    else
    {
      // vins跟丢了, 准备重启
      // ROS_WARN("VINS failure detected.");
      lastVinsTransAvailable = false;
      odomResetId = cloudInfo.odomResetId;
    }

    // use imu incremental estimation for pose guess (only rotation)
    // 3.vins odom无法使用, 只能使用imu来计算点云的先验 (只更新rpy)
    if (cloudInfo.imuAvailable == true)
    {
      //这个是与初始化的那一步一样 纯靠imu原始数据
        //只更新旋转量 步骤和上面的一致 
        //其实这里没太理解用两个方法进行估计的优势在哪  
        //同时大家需要关注transformTobeMapped这一项 后续十分关键
      //在imageProjection中根据imuPreintegration里发布的odom/imu_incremental里程计话题获得的初始估计
      // ROS_INFO("Using IMU initial guess");
      //对应的订阅话题   这一行是image那个文件里的 subOdom = nh.subscribe<nav_msgs::Odometry>(odomTopic+"_incremental", 2000, &ImageProjection::odometryHandler, this, ros::TransportHints().tcpNoDelay());
      Eigen::Affine3f transBack = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit);
      //上一次变换的逆乘此次的变换 获得增量
      Eigen::Affine3f transIncre = lastImuTransformation.inverse() * transBack;
       //最终的变换 利用该变换给transformTobeMapped赋值
      Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);
      Eigen::Affine3f transFinal = transTobe * transIncre;
      pcl::getTranslationAndEulerAngles(transFinal, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
      //更新最后的变换值
      lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit);  // save imu before return;
      return;
    }
  }
  
  //1.2.1
  // 提取附近的keyframes及其点云, 来构造localmap
  void extractNearby()
  {
    // 附近的keyframes (最后一个keyframe附近, 50m)
    //定义周围关键帧点云指针以及降采样后指针  
    pcl::PointCloud<PointType>::Ptr surroundingKeyPoses(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr surroundingKeyPosesDS(new pcl::PointCloud<PointType>());
    std::vector<int> pointSearchInd;
    std::vector<float> pointSearchSqDis;

    // extract all the nearby key poses and downsample them
    //提取周围的所有关键帧并降采样
    kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D);  // create kd-tree创建Kd树然后搜索  半径在配置文件中
    kdtreeSurroundingKeyPoses->radiusSearch(cloudKeyPoses3D->back(), (double) surroundingKeyframeSearchRadius, pointSearchInd, pointSearchSqDis);
    for (int i = 0; i < (int) pointSearchInd.size(); ++i)
    {
      int id = pointSearchInd[i];//保存附近关键帧
      surroundingKeyPoses->push_back(cloudKeyPoses3D->points[id]);
    }
    //降采样
    downSizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);
    downSizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDS);

    // also extract some latest key frames in case the robot rotates in one position
    //提取了一些最新的关键帧，以防机器人在一个位置原地旋转 
    int numPoses = cloudKeyPoses3D->size();
    for (int i = numPoses - 1; i >= 0; --i)
    {//最近十秒内的关键帧
      if (timeLaserInfoCur - cloudKeyPoses6D->points[i].time < 10.0)
        surroundingKeyPosesDS->push_back(cloudKeyPoses3D->points[i]);
      else
        break;
    }
    //对降采样后的点云进行提取出边缘点和平面点对应的localmap 
    extractCloud(surroundingKeyPosesDS);
  }
  //1.2.1.1
  // 通过提取到的keyframes, 来提取点云, 从而构造localmap
  void extractCloud(pcl::PointCloud<PointType>::Ptr cloudToExtract)
  {
    //激光点云 关键帧周围角点 容器
    std::vector<pcl::PointCloud<PointType>> laserCloudCornerSurroundingVec;
    std::vector<pcl::PointCloud<PointType>> laserCloudSurfSurroundingVec;

    laserCloudCornerSurroundingVec.resize(cloudToExtract->size());
    laserCloudSurfSurroundingVec.resize(cloudToExtract->size());

    // extract surrounding map提取周边地图
    #pragma omp parallel for num_threads(numberOfCores)//线程
    for (int i = 0; i < (int) cloudToExtract->size(); ++i)
    {
        //将强度存到thisKeyInd中
      int thisKeyInd = (int) cloudToExtract->points[i].intensity;
       //如果待提取的点云中某点与关键帧点云back距离超过搜索半径则不执行后续的 再进行for的下一次操作
      if (pointDistance(cloudKeyPoses3D->points[thisKeyInd], cloudKeyPoses3D->back()) > surroundingKeyframeSearchRadius)
        continue;
      laserCloudCornerSurroundingVec[i] = *transformPointCloud(cornerCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
      laserCloudSurfSurroundingVec[i] = *transformPointCloud(surfCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
    }

    // fuse the map 保险
    laserCloudCornerFromMap->clear();
    laserCloudSurfFromMap->clear();
    //对于角点容器进行叠加，到local map大小
    for (int i = 0; i < (int) cloudToExtract->size(); ++i)
    {
      *laserCloudCornerFromMap += laserCloudCornerSurroundingVec[i];
      *laserCloudSurfFromMap += laserCloudSurfSurroundingVec[i];
    }
    
    //降采样关键帧点云************************************
    // Downsample the surrounding corner key frames (or map)
    downSizeFilterCorner.setInputCloud(laserCloudCornerFromMap);
    downSizeFilterCorner.filter(*laserCloudCornerFromMapDS);
    // Downsample the surrounding surf key frames (or map)
    downSizeFilterSurf.setInputCloud(laserCloudSurfFromMap);
    downSizeFilterSurf.filter(*laserCloudSurfFromMapDS);
  }
  //1.2
  // 提取附近的keyframes及其点云, 来构造localmap
  void extractSurroundingKeyFrames()
  {
    if (cloudKeyPoses3D->points.empty() == true)
      return;
    //以下是作者注释掉的 没有区分 直接按提取周围关键帧来
        // if (loopClosureEnableFlag == true)
        // {
        //     extractForLoopClosure();    
        // } else {
        //     extractNearby();
        // }

       //提取周围的关键帧
    extractNearby();
  }

  //降采样
  void downsampleCurrentScan()
  {
    // Downsample cloud from current scan
    //对当前帧点云降采样  刚刚完成了周围关键帧的降采样  
    //大量的降采样工作无非是为了使点云稀疏化 加快匹配以及实时性要求
    laserCloudCornerLastDS->clear();
    downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
    downSizeFilterCorner.filter(*laserCloudCornerLastDS);//滤波
    laserCloudCornerLastDSNum = laserCloudCornerLastDS->size();//说明大小

    laserCloudSurfLastDS->clear();
    downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
    downSizeFilterSurf.filter(*laserCloudSurfLastDS);
    laserCloudSurfLastDSNum = laserCloudSurfLastDS->size();
  }
   //1.4.1.1
  // 更新当前帧的位姿 (lidar到map的坐标变换transform)
  void updatePointAssociateToMap()
  {
    transPointAssociateToMap = trans2Affine3f(transformTobeMapped);  //
  }
  
  //1.4.1
  // 构造 点到直线 的残差约束 (并行计算)
  void cornerOptimization()
  {  //实现transformTobeMapped的矩阵形式转换 下面调用的函数就一行就不展开了  工具类函数
    updatePointAssociateToMap();

#pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < laserCloudCornerLastDSNum; i++)
    {
      PointType pointOri, pointSel, coeff;
      std::vector<int> pointSearchInd;
      std::vector<float> pointSearchSqDis;

      pointOri = laserCloudCornerLastDS->points[i];
      pointAssociateToMap(&pointOri, &pointSel);
      kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

      cv::Mat matA1(3, 3, CV_32F, cv::Scalar::all(0));
      cv::Mat matD1(1, 3, CV_32F, cv::Scalar::all(0));
      cv::Mat matV1(3, 3, CV_32F, cv::Scalar::all(0));

      if (pointSearchSqDis[4] < 1.0)
      {
        float cx = 0, cy = 0, cz = 0;
        for (int j = 0; j < 5; j++)
        {
          cx += laserCloudCornerFromMapDS->points[pointSearchInd[j]].x;
          cy += laserCloudCornerFromMapDS->points[pointSearchInd[j]].y;
          cz += laserCloudCornerFromMapDS->points[pointSearchInd[j]].z;
        }
        cx /= 5;
        cy /= 5;
        cz /= 5;

        float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
        for (int j = 0; j < 5; j++)
        {
          float ax = laserCloudCornerFromMapDS->points[pointSearchInd[j]].x - cx;
          float ay = laserCloudCornerFromMapDS->points[pointSearchInd[j]].y - cy;
          float az = laserCloudCornerFromMapDS->points[pointSearchInd[j]].z - cz;

          a11 += ax * ax;
          a12 += ax * ay;
          a13 += ax * az;
          a22 += ay * ay;
          a23 += ay * az;
          a33 += az * az;
        }
        a11 /= 5;
        a12 /= 5;
        a13 /= 5;
        a22 /= 5;
        a23 /= 5;
        a33 /= 5;

        matA1.at<float>(0, 0) = a11;
        matA1.at<float>(0, 1) = a12;
        matA1.at<float>(0, 2) = a13;
        matA1.at<float>(1, 0) = a12;
        matA1.at<float>(1, 1) = a22;
        matA1.at<float>(1, 2) = a23;
        matA1.at<float>(2, 0) = a13;
        matA1.at<float>(2, 1) = a23;
        matA1.at<float>(2, 2) = a33;

        // 协方差矩阵与点云中角点面点之间的关系:
        // 1.假设点云序列为S，计算 S 的协方差矩阵，记为 cov_mat ，cov_mat 的特征值记为 V ，特征向量记为 E 。
        // 2.如果 S 分布在一条线段上，那么 V 中一个特征值就会明显比其他两个大，E 中与较大特征值相对应的特征向量代表边缘线的方向。(一大两小，大的代表直线方向)
        // 3.如果 S 分布在一块平面上，那么 V 中一个特征值就会明显比其他两个小，E 中与较小特征值相对应的特征向量代表平面片的方向。(一小两大，小方向)边缘线或平面块的位置通过穿过 S 的几何中心来确定。

        // 计算协方差矩阵的特征值和特征向量，用于判断这5个点是不是呈线状分布，此为PCA的原理；
        cv::eigen(matA1, matD1, matV1);

        // 如果5个点呈线状分布，最大的特征值对应的特征向量就是该线的方向向量；
        if (matD1.at<float>(0, 0) > 3 * matD1.at<float>(0, 1))
        {
          float x0 = pointSel.x;
          float y0 = pointSel.y;
          float z0 = pointSel.z;
          // 从中心点沿着方向向量向两端移动0.1m，构造线上的两个点；
          float x1 = cx + 0.1 * matV1.at<float>(0, 0);
          float y1 = cy + 0.1 * matV1.at<float>(0, 1);
          float z1 = cz + 0.1 * matV1.at<float>(0, 2);
          float x2 = cx - 0.1 * matV1.at<float>(0, 0);
          float y2 = cy - 0.1 * matV1.at<float>(0, 1);
          float z2 = cz - 0.1 * matV1.at<float>(0, 2);

          // 向量OA = (x0 - x1, y0 - y1, z0 - z1), 向量OB = (x0 - x2, y0 - y2, z0 - z2)，向量AB = （x1 - x2, y1 - y2, z1 - z2）;
          // 点到线的距离，d = |向量OA 叉乘 向量OB|/|AB|;
          // 向量OA 叉乘 向量OB 得到的向量模长 ： 是垂直a、b所在平面，且以|b|·sinθ为高、|a|为底的平行四边形的面积，
          // 因此|向量OA 叉乘 向量OB|再除以|AB|的模长，则得到高度，即点到线的距离；

          float a012 = sqrt(((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) + ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) + ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1)) * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1)));

          float l12 = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2));

          float la = ((y1 - y2) * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) + (z1 - z2) * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1))) / a012 / l12;

          float lb = -((x1 - x2) * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) - (z1 - z2) * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1))) / a012 / l12;

          float lc = -((x1 - x2) * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) + (y1 - y2) * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1))) / a012 / l12;

          float ld2 = a012 / l12;

          float s = 1 - 0.9 * fabs(ld2);


          coeff.x = s * la;
          coeff.y = s * lb;
          coeff.z = s * lc;
          coeff.intensity = s * ld2;

          if (s > 0.1)
          {
            laserCloudOriCornerVec[i] = pointOri;
            coeffSelCornerVec[i] = coeff;
            laserCloudOriCornerFlag[i] = true;
          }
        }
      }
    }
  }
  //1.4.2
  // 构建 点到平面 的残差约束
  void surfOptimization()
  {
    updatePointAssociateToMap();

#pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < laserCloudSurfLastDSNum; i++)
    {
      PointType pointOri, pointSel, coeff;
      std::vector<int> pointSearchInd;
      std::vector<float> pointSearchSqDis;

      pointOri = laserCloudSurfLastDS->points[i];
      pointAssociateToMap(&pointOri, &pointSel);
      kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

      Eigen::Matrix<float, 5, 3> matA0;
      Eigen::Matrix<float, 5, 1> matB0;
      Eigen::Vector3f matX0;

      matA0.setZero();
      matB0.fill(-1);
      matX0.setZero();

      if (pointSearchSqDis[4] < 1.0)
      {
        // 求面的法向量不是用的PCA，使用的是最小二乘拟合；
        // 假设平面不通过原点，则平面的一般方程为Ax + By + Cz + 1 = 0，用这个假设可以少算一个参数；
        for (int j = 0; j < 5; j++)
        {
          matA0(j, 0) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].x;
          matA0(j, 1) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].y;
          matA0(j, 2) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].z;
        }

        // 构建超定方程组： matA0 * norm（A, B, C） = matB0；

        // 求解这个最小二乘问题，可得平面的法向量norm（A, B, C）；
        matX0 = matA0.colPivHouseholderQr().solve(matB0);

        float pa = matX0(0, 0);
        float pb = matX0(1, 0);
        float pc = matX0(2, 0);
        float pd = 1;

        // Ax + By + Cz + 1 = 0，全部除以法向量的模长，方程依旧成立，而且使得法向量归一化了；
        float ps = sqrt(pa * pa + pb * pb + pc * pc);
        pa /= ps;
        pb /= ps;
        pc /= ps;
        pd /= ps;

        // 点(x0, y0, z0)到平面Ax + By + Cz + D = 0 的距离 = fabs(A*x0 + B*y0 + C*z0 + D) / sqrt(A^2 + B^2 + C^2)；
        // 因为法向量（A, B, C）已经归一化了，所以距离公式可以简写为：距离 = fabs(A*x0 + B*y0 + C*z0 + D) ；

        bool planeValid = true;
        for (int j = 0; j < 5; j++)
        {
          // 如果拟合的５个面点中，任何一个点到平面的距离大于阈值，则认为平面拟合不好；
          if (fabs(pa * laserCloudSurfFromMapDS->points[pointSearchInd[j]].x + pb * laserCloudSurfFromMapDS->points[pointSearchInd[j]].y + pc * laserCloudSurfFromMapDS->points[pointSearchInd[j]].z + pd) > 0.2)
          {
            planeValid = false;
            break;
          }
        }

        if (planeValid)
        {
          float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;

          float s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(pointSel.x * pointSel.x + pointSel.y * pointSel.y + pointSel.z * pointSel.z));

          coeff.x = s * pa;
          coeff.y = s * pb;
          coeff.z = s * pc;
          coeff.intensity = s * pd2;

          if (s > 0.1)
          {
            laserCloudOriSurfVec[i] = pointOri;
            coeffSelSurfVec[i] = coeff;
            laserCloudOriSurfFlag[i] = true;
          }
        }
      }
    }
  }
  //1.4.3
  // 联合两类残差 (点到直线, 点到平面)
  void combineOptimizationCoeffs()
    {
    // combine corner coeffs
    for (int i = 0; i < laserCloudCornerLastDSNum; ++i)
    {
      if (laserCloudOriCornerFlag[i] == true)
      {
        laserCloudOri->push_back(laserCloudOriCornerVec[i]);
        coeffSel->push_back(coeffSelCornerVec[i]);
      }
    }
    // combine surf coeffs
    for (int i = 0; i < laserCloudSurfLastDSNum; ++i)
    {
      if (laserCloudOriSurfFlag[i] == true)
      {
        laserCloudOri->push_back(laserCloudOriSurfVec[i]);
        coeffSel->push_back(coeffSelSurfVec[i]);
      }
    }
    // reset flag for next iteration
    std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
    std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);
  }

  bool LMOptimization(int iterCount)
  {
    // This optimization is from the original loam_velodyne, need to cope with coordinate transformation
    // lidar <- camera      ---     camera <- lidar
    // x = z                ---     x = y
    // y = x                ---     y = z
    // z = y                ---     z = x
    // roll = yaw           ---     roll = pitch
    // pitch = roll         ---     pitch = yaw
    // yaw = pitch          ---     yaw = roll

    // lidar -> camera
    float srx = sin(transformTobeMapped[1]);
    float crx = cos(transformTobeMapped[1]);
    float sry = sin(transformTobeMapped[2]);
    float cry = cos(transformTobeMapped[2]);
    float srz = sin(transformTobeMapped[0]);
    float crz = cos(transformTobeMapped[0]);

    int laserCloudSelNum = laserCloudOri->size();
    if (laserCloudSelNum < 50)
    {
      return false;
    }

    cv::Mat matA(laserCloudSelNum, 6, CV_32F, cv::Scalar::all(0));
    cv::Mat matAt(6, laserCloudSelNum, CV_32F, cv::Scalar::all(0));
    cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
    cv::Mat matB(laserCloudSelNum, 1, CV_32F, cv::Scalar::all(0));
    cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
    cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));

    PointType pointOri, coeff;

    for (int i = 0; i < laserCloudSelNum; i++)
    {
      // lidar -> camera
      pointOri.x = laserCloudOri->points[i].y;
      pointOri.y = laserCloudOri->points[i].z;
      pointOri.z = laserCloudOri->points[i].x;
      // lidar -> camera
      coeff.x = coeffSel->points[i].y;
      coeff.y = coeffSel->points[i].z;
      coeff.z = coeffSel->points[i].x;
      coeff.intensity = coeffSel->points[i].intensity;
      // in camera
      float arx = (crx * sry * srz * pointOri.x + crx * crz * sry * pointOri.y - srx * sry * pointOri.z) * coeff.x + (-srx * srz * pointOri.x - crz * srx * pointOri.y - crx * pointOri.z) * coeff.y + (crx * cry * srz * pointOri.x + crx * cry * crz * pointOri.y - cry * srx * pointOri.z) * coeff.z;

      float ary = ((cry * srx * srz - crz * sry) * pointOri.x + (sry * srz + cry * crz * srx) * pointOri.y + crx * cry * pointOri.z) * coeff.x + ((-cry * crz - srx * sry * srz) * pointOri.x + (cry * srz - crz * srx * sry) * pointOri.y - crx * sry * pointOri.z) * coeff.z;

      float arz = ((crz * srx * sry - cry * srz) * pointOri.x + (-cry * crz - srx * sry * srz) * pointOri.y) * coeff.x + (crx * crz * pointOri.x - crx * srz * pointOri.y) * coeff.y + ((sry * srz + cry * crz * srx) * pointOri.x + (crz * sry - cry * srx * srz) * pointOri.y) * coeff.z;
      // lidar -> camera
      matA.at<float>(i, 0) = arz;
      matA.at<float>(i, 1) = arx;
      matA.at<float>(i, 2) = ary;
      //坐标系转换
      matA.at<float>(i, 3) = coeff.z;//coeff带有s信息矩阵
      matA.at<float>(i, 4) = coeff.x;
      matA.at<float>(i, 5) = coeff.y;
      matB.at<float>(i, 0) = -coeff.intensity;
    }
    //matAtA = J^T *∑^-1 *J     matAtB=J^T *∑^-1*e
    cv::transpose(matA, matAt);
    matAtA = matAt * matA;
    matAtB = matAt * matB;
    cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);//DECOMP_QR分解求解方程

    if (iterCount == 0)
    {
      cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
      cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
      cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));

      // 对AtA进行特征分解
      cv::eigen(matAtA, matE, matV);
      matV.copyTo(matV2);

      isDegenerate = false;
      float eignThre[6] = {100, 100, 100, 100, 100, 100};
      for (int i = 5; i >= 0; i--)
      {
        if (matE.at<float>(0, i) < eignThre[i])
        {
          for (int j = 0; j < 6; j++)
          {
            matV2.at<float>(i, j) = 0;  //
          }
          // 点云退化了
          isDegenerate = true;
        }
        else
        {
          break;
        }
      }
      matP = matV.inv() * matV2;
    }

    // 点云退化了
    if (isDegenerate)
    {
      cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
      matX.copyTo(matX2);
      matX = matP * matX2;
    }

    transformTobeMapped[0] += matX.at<float>(0, 0);
    transformTobeMapped[1] += matX.at<float>(1, 0);
    transformTobeMapped[2] += matX.at<float>(2, 0);
    transformTobeMapped[3] += matX.at<float>(3, 0);
    transformTobeMapped[4] += matX.at<float>(4, 0);
    transformTobeMapped[5] += matX.at<float>(5, 0);

    float deltaR = sqrt(pow(pcl::rad2deg(matX.at<float>(0, 0)), 2) + pow(pcl::rad2deg(matX.at<float>(1, 0)), 2) + pow(pcl::rad2deg(matX.at<float>(2, 0)), 2));
    float deltaT = sqrt(pow(matX.at<float>(3, 0) * 100, 2) + pow(matX.at<float>(4, 0) * 100, 2) + pow(matX.at<float>(5, 0) * 100, 2));

    if (deltaR < 0.05 && deltaT < 0.05)
    {
      return true;  // converged
    }
    return false;  // keep optimizing
  }
  
  //1.4 
      //扫描到地图的匹配与优化  这里与LEGO差不多思想
      //它分为角点优化、平面点优化、配准与更新等部分。
      //根据现有地图与最新点云数据进行配准从而更新机器人精确位姿与融合建图，
      //优化的过程与里程计的计算类似，是通过计算点到直线或平面的距离，构建优化公式再用LM法求解。
  void scan2MapOptimization()
  {
    if (cloudKeyPoses3D->points.empty())
      return;

    if (laserCloudCornerLastDSNum > edgeFeatureMinValidNum && laserCloudSurfLastDSNum > surfFeatureMinValidNum)
    {//构建kdtree
      kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMapDS);
      kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMapDS);
      //迭代30次
      for (int iterCount = 0; iterCount < 30; iterCount++)
      {
        laserCloudOri->clear();
        coeffSel->clear();
        //边缘点匹配优化
        cornerOptimization();
        //平面点匹配优化
        surfOptimization();
        //组合优化多项式系数
        combineOptimizationCoeffs();
         //使用了9轴imu的orientation与做transformTobeMapped插值，并且roll和pitch收到常量阈值约束（权重）
        if (LMOptimization(iterCount) == true)
          break;
      }
      //使用了9轴imu的方向与做transformTobeMapped插值，并且roll和pitch收到常量阈值约束（权重）
      transformUpdate();
    }
    else
    {
      ROS_WARN("Not enough features! Only %d edge and %d planar features available.", laserCloudCornerLastDSNum, laserCloudSurfLastDSNum);
    }
  }

  void transformUpdate()
  {
    if (cloudInfo.imuAvailable == true)
    {
      if (std::abs(cloudInfo.imuPitchInit) < 1.4)
      {
        double imuWeight = 0.01;
        tf::Quaternion imuQuaternion;
        tf::Quaternion transformQuaternion;
        double rollMid, pitchMid, yawMid;

        // slerp roll
        transformQuaternion.setRPY(transformTobeMapped[0], 0, 0);
        imuQuaternion.setRPY(cloudInfo.imuRollInit, 0, 0);
        tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
        transformTobeMapped[0] = rollMid;

        // slerp pitch
        transformQuaternion.setRPY(0, transformTobeMapped[1], 0);
        imuQuaternion.setRPY(0, cloudInfo.imuPitchInit, 0);
        tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
        transformTobeMapped[1] = pitchMid;
      }
    }

    transformTobeMapped[0] = constraintTransformation(transformTobeMapped[0], rotation_tollerance);
    transformTobeMapped[1] = constraintTransformation(transformTobeMapped[1], rotation_tollerance);
    transformTobeMapped[5] = constraintTransformation(transformTobeMapped[5], z_tollerance);
  }

  float constraintTransformation(float value, float limit)
  {
    if (value < -limit)
      value = -limit;
    if (value > limit)
      value = limit;

    return value;
  }

  // 是否将当前帧设为关键帧
  bool saveFrame()
  {
    if (cloudKeyPoses3D->points.empty())
      return true;

    Eigen::Affine3f transStart = pclPointToAffine3f(cloudKeyPoses6D->back());
    Eigen::Affine3f transFinal = pcl::getTransformation(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
    Eigen::Affine3f transBetween = transStart.inverse() * transFinal;
    float x, y, z, roll, pitch, yaw;
    pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw);

    if (abs(roll) < surroundingkeyframeAddingAngleThreshold && abs(pitch) < surroundingkeyframeAddingAngleThreshold && abs(yaw) < surroundingkeyframeAddingAngleThreshold && sqrt(x * x + y * y + z * z) < surroundingkeyframeAddingDistThreshold)
      return false;

    return true;
  }

  void addOdomFactor()
  {
    if (cloudKeyPoses3D->points.empty())
    {
      noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-2, 1e-2, M_PI * M_PI, 1e8, 1e8, 1e8).finished());  // rad*rad, meter*meter
      gtSAMgraph.add(PriorFactor<Pose3>(0, trans2gtsamPose(transformTobeMapped), priorNoise));
      initialEstimate.insert(0, trans2gtsamPose(transformTobeMapped));
    }
    else
    {
      noiseModel::Diagonal::shared_ptr odometryNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
      gtsam::Pose3 poseFrom = pclPointTogtsamPose3(cloudKeyPoses6D->points.back());
      gtsam::Pose3 poseTo = trans2gtsamPose(transformTobeMapped);
      gtSAMgraph.add(BetweenFactor<Pose3>(cloudKeyPoses3D->size() - 1, cloudKeyPoses3D->size(), poseFrom.between(poseTo), odometryNoise));
      initialEstimate.insert(cloudKeyPoses3D->size(), poseTo);
      // if (isDegenerate)
      // {
      // adding VINS constraints is deleted as benefits are not obvious, disable for now
      // gtSAMgraph.add(BetweenFactor<Pose3>(cloudKeyPoses3D->size()-1, cloudKeyPoses3D->size(), vinsPoseFrom.between(vinsPoseTo), odometryNoise));
      // }
    }
  }

  void addGPSFactor()
  {
    if (gpsQueue.empty())
      return;

    // wait for system initialized and settles down
    if (cloudKeyPoses3D->points.empty())
      return;
    else if (pointDistance(cloudKeyPoses3D->front(), cloudKeyPoses3D->back()) < 5.0)
      return;

    // pose covariance small, no need to correct
    if (poseCovariance(3, 3) < poseCovThreshold && poseCovariance(4, 4) < poseCovThreshold)
      return;

    // last gps position
    static PointType lastGPSPoint;

    while (!gpsQueue.empty())
    {
      if (gpsQueue.front().header.stamp.toSec() < timeLaserInfoCur - 0.2)
      {
        // message too old
        gpsQueue.pop_front();
      }
      else if (gpsQueue.front().header.stamp.toSec() > timeLaserInfoCur + 0.2)
      {
        // message too new
        break;
      }
      else
      {
        nav_msgs::Odometry thisGPS = gpsQueue.front();
        gpsQueue.pop_front();

        // GPS too noisy, skip
        float noise_x = thisGPS.pose.covariance[0];
        float noise_y = thisGPS.pose.covariance[7];
        float noise_z = thisGPS.pose.covariance[14];
        if (noise_x > gpsCovThreshold || noise_y > gpsCovThreshold)
          continue;

        float gps_x = thisGPS.pose.pose.position.x;
        float gps_y = thisGPS.pose.pose.position.y;
        float gps_z = thisGPS.pose.pose.position.z;
        if (!useGpsElevation)
        {
          gps_z = transformTobeMapped[5];
          noise_z = 0.01;
        }

        // GPS not properly initialized (0,0,0)
        if (abs(gps_x) < 1e-6 && abs(gps_y) < 1e-6)
          continue;

        // Add GPS every a few meters
        PointType curGPSPoint;
        curGPSPoint.x = gps_x;
        curGPSPoint.y = gps_y;
        curGPSPoint.z = gps_z;
        if (pointDistance(curGPSPoint, lastGPSPoint) < 5.0)
          continue;
        else
          lastGPSPoint = curGPSPoint;

        gtsam::Vector Vector3(3);
        Vector3 << max(noise_x, 1.0f), max(noise_y, 1.0f), max(noise_z, 1.0f);
        noiseModel::Diagonal::shared_ptr gps_noise = noiseModel::Diagonal::Variances(Vector3);
        gtsam::GPSFactor gps_factor(cloudKeyPoses3D->size(), gtsam::Point3(gps_x, gps_y, gps_z), gps_noise);
        gtSAMgraph.add(gps_factor);

        aLoopIsClosed = true;

        break;
      }
    }
  }

  void addLoopFactor()
  {
    if (loopIndexQueue.empty())
      return;

    for (size_t i = 0; i < loopIndexQueue.size(); ++i)
    {
      int indexFrom = loopIndexQueue[i].first;
      int indexTo = loopIndexQueue[i].second;
      gtsam::Pose3 poseBetween = loopPoseQueue[i];
      gtsam::noiseModel::Diagonal::shared_ptr noiseBetween = loopNoiseQueue[i];
      gtSAMgraph.add(BetweenFactor<Pose3>(indexFrom, indexTo, poseBetween, noiseBetween));
    }

    loopIndexQueue.clear();
    loopPoseQueue.clear();
    loopNoiseQueue.clear();
    aLoopIsClosed = true;
  }

  void saveKeyFramesAndFactor()
  {
    if (saveFrame() == false)
      return;

    // odom factor
    addOdomFactor();

    // gps factor
    // addGPSFactor();

    // loop factor
    addLoopFactor();

    // update iSAM
    isam->update(gtSAMgraph, initialEstimate);
    isam->update();

    gtSAMgraph.resize(0);
    initialEstimate.clear();

    // save key poses
    PointType thisPose3D;
    PointTypePose thisPose6D;
    Pose3 latestEstimate;

    isamCurrentEstimate = isam->calculateEstimate();
    latestEstimate = isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size() - 1);
    // cout << "****************************************************" << endl;
    // isamCurrentEstimate.print("Current estimate: ");

    thisPose3D.x = latestEstimate.translation().x();
    thisPose3D.y = latestEstimate.translation().y();
    thisPose3D.z = latestEstimate.translation().z();
    thisPose3D.intensity = cloudKeyPoses3D->size();  // this can be used as index
    cloudKeyPoses3D->push_back(thisPose3D);

    thisPose6D.x = thisPose3D.x;
    thisPose6D.y = thisPose3D.y;
    thisPose6D.z = thisPose3D.z;
    thisPose6D.intensity = thisPose3D.intensity;  // this can be used as index
    thisPose6D.roll = latestEstimate.rotation().roll();
    thisPose6D.pitch = latestEstimate.rotation().pitch();
    thisPose6D.yaw = latestEstimate.rotation().yaw();
    thisPose6D.time = timeLaserInfoCur;
    cloudKeyPoses6D->push_back(thisPose6D);

    // cout << "****************************************************" << endl;
    // cout << "Pose covariance:" << endl;
    // cout << isam->marginalCovariance(isamCurrentEstimate.size()-1) << endl << endl;
    poseCovariance = isam->marginalCovariance(isamCurrentEstimate.size() - 1);

    // save updated transform
    transformTobeMapped[0] = latestEstimate.rotation().roll();
    transformTobeMapped[1] = latestEstimate.rotation().pitch();
    transformTobeMapped[2] = latestEstimate.rotation().yaw();
    transformTobeMapped[3] = latestEstimate.translation().x();
    transformTobeMapped[4] = latestEstimate.translation().y();
    transformTobeMapped[5] = latestEstimate.translation().z();

    // save all the received edge and surf points
    pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
    pcl::copyPointCloud(*laserCloudCornerLastDS, *thisCornerKeyFrame);
    pcl::copyPointCloud(*laserCloudSurfLastDS, *thisSurfKeyFrame);

    // save key frame cloud
    cornerCloudKeyFrames.push_back(thisCornerKeyFrame);
    surfCloudKeyFrames.push_back(thisSurfKeyFrame);

    // save path for visualization
    updatePath(thisPose6D);
  }

  void correctPoses()
  {
    if (cloudKeyPoses3D->points.empty())
      return;

    if (aLoopIsClosed == true)
    {
      // clear path
      globalPath.poses.clear();

      // update key poses
      int numPoses = isamCurrentEstimate.size();
      for (int i = 0; i < numPoses; ++i)
      {
        cloudKeyPoses3D->points[i].x = isamCurrentEstimate.at<Pose3>(i).translation().x();
        cloudKeyPoses3D->points[i].y = isamCurrentEstimate.at<Pose3>(i).translation().y();
        cloudKeyPoses3D->points[i].z = isamCurrentEstimate.at<Pose3>(i).translation().z();

        cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
        cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
        cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
        cloudKeyPoses6D->points[i].roll = isamCurrentEstimate.at<Pose3>(i).rotation().roll();
        cloudKeyPoses6D->points[i].pitch = isamCurrentEstimate.at<Pose3>(i).rotation().pitch();
        cloudKeyPoses6D->points[i].yaw = isamCurrentEstimate.at<Pose3>(i).rotation().yaw();

        updatePath(cloudKeyPoses6D->points[i]);
      }

      aLoopIsClosed = false;
      // ID for reseting IMU pre-integration
      ++imuPreintegrationResetId;
    }
  }

  void publishOdometry()
  {
    // Publish odometry for ROS
    nav_msgs::Odometry laserOdometryROS;
    laserOdometryROS.header.stamp = timeLaserInfoStamp;
    laserOdometryROS.header.frame_id = "odom";
    laserOdometryROS.child_frame_id = "odom_mapping";
    laserOdometryROS.pose.pose.position.x = transformTobeMapped[3];
    laserOdometryROS.pose.pose.position.y = transformTobeMapped[4];
    laserOdometryROS.pose.pose.position.z = transformTobeMapped[5];
    laserOdometryROS.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
    laserOdometryROS.pose.covariance[0] = double(imuPreintegrationResetId);
    pubOdomAftMappedROS.publish(laserOdometryROS);
    // Publish TF
    static tf::TransformBroadcaster br;
    tf::Transform t_odom_to_lidar = tf::Transform(tf::createQuaternionFromRPY(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]), tf::Vector3(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5]));
    tf::StampedTransform trans_odom_to_lidar = tf::StampedTransform(t_odom_to_lidar, timeLaserInfoStamp, "odom", "lidar_link");
    br.sendTransform(trans_odom_to_lidar);
  }

  void updatePath(const PointTypePose &pose_in)
  {
    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header.stamp = ros::Time().fromSec(pose_in.time);
    pose_stamped.header.frame_id = "odom";
    pose_stamped.pose.position.x = pose_in.x;
    pose_stamped.pose.position.y = pose_in.y;
    pose_stamped.pose.position.z = pose_in.z;
    tf::Quaternion q = tf::createQuaternionFromRPY(pose_in.roll, pose_in.pitch, pose_in.yaw);
    pose_stamped.pose.orientation.x = q.x();
    pose_stamped.pose.orientation.y = q.y();
    pose_stamped.pose.orientation.z = q.z();
    pose_stamped.pose.orientation.w = q.w();

    globalPath.poses.push_back(pose_stamped);
  }

  void publishFrames()
  {
    if (cloudKeyPoses3D->points.empty())
      return;
    // publish key poses
    publishCloud(&pubKeyPoses, cloudKeyPoses6D, timeLaserInfoStamp, "odom");
    // Publish surrounding key frames
    publishCloud(&pubRecentKeyFrames, laserCloudSurfFromMapDS, timeLaserInfoStamp, "odom");
    // publish registered key frame
    if (pubRecentKeyFrame.getNumSubscribers() != 0)
    {
      pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
      PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
      *cloudOut += *transformPointCloud(laserCloudCornerLastDS, &thisPose6D);
      *cloudOut += *transformPointCloud(laserCloudSurfLastDS, &thisPose6D);
      publishCloud(&pubRecentKeyFrame, cloudOut, timeLaserInfoStamp, "odom");
    }
    // publish registered high-res raw cloud
    if (pubCloudRegisteredRaw.getNumSubscribers() != 0)
    {
      pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
      pcl::fromROSMsg(cloudInfo.cloud_deskewed, *cloudOut);
      PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
      *cloudOut = *transformPointCloud(cloudOut, &thisPose6D);
      publishCloud(&pubCloudRegisteredRaw, cloudOut, timeLaserInfoStamp, "odom");
    }
    // publish path
    if (pubPath.getNumSubscribers() != 0)
    {
      globalPath.header.stamp = timeLaserInfoStamp;
      globalPath.header.frame_id = "odom";
      pubPath.publish(globalPath);
    }
  }
};

//高斯牛顿手写法+欧拉角表示
int main(int argc, char **argv)
{
  ros::init(argc, argv, "lidar");

  mapOptimization MO;//定义类，调用构造函数

  ROS_INFO("\033[1;32m----> Lidar Map Optimization Started.\033[0m");

  std::thread loopDetectionthread(&mapOptimization::loopClosureThread, &MO);
  std::thread visualizeMapThread(&mapOptimization::visualizeGlobalMapThread, &MO);

  ros::spin();

  loopDetectionthread.join();
  visualizeMapThread.join();

  return 0;
}