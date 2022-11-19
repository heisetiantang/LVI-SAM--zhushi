#include "utility.h"

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
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>

using gtsam::symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)，Pose3姿态 (x,y,z,r,p,y)
using gtsam::symbol_shorthand::V; // Vel   (xdot,ydot,zdot)，Vel速度导数   (xdot,ydot,zdot)
using gtsam::symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)，Bias陀螺仪残差  (ax,ay,az,gx,gy,gz)

//Imu预积分
class IMUPreintegration : public ParamServer
{
public:

    ros::Subscriber subImu;//接收imu话题
    ros::Subscriber subOdometry;//订阅mapoptimization位姿信息,
    ros::Publisher pubImuOdometry;//发布IMU里程计（IMU频率）来自于帧间约束后的
    ros::Publisher pubImuPath;//发布path(0.1s)

    // map -> odom
    tf::Transform map_to_odom;
    tf::TransformBroadcaster tfMap2Odom;
    // odom -> base_link
    tf::TransformBroadcaster tfOdom2BaseLink;

    bool systemInitialized = false;
   
   
   
    // 噪声协方差
    gtsam::noiseModel::Diagonal::shared_ptr priorPoseNoise;//位姿噪声协方差
    gtsam::noiseModel::Diagonal::shared_ptr priorVelNoise;//速度噪声协方差
    gtsam::noiseModel::Diagonal::shared_ptr priorBiasNoise;//Bias
    gtsam::noiseModel::Diagonal::shared_ptr correctionNoise;
    gtsam::Vector noiseModelBetweenBias;

    // imu 预积分器
    //imuIntegratorOpt_负责预积分两个激光里程计之间的imu数据，作为约束加入因子图，并且优化出bias
    gtsam::PreintegratedImuMeasurements *imuIntegratorOpt_;
    //imuIntegratorImu_用来根据新的激光里程计到达后已经优化好的bias，预测从当前帧开始，下一帧激光里程计到达之前的imu里程计增量
    gtsam::PreintegratedImuMeasurements *imuIntegratorImu_;

    // imu 数据队列buffer
    //imuQueOpt用来给imuIntegratorOpt_提供数据来源，不要的就弹出(从队头开始出发，比当前激光里程计数据早的imu通通积分，用一个扔一个)；
    std::deque<sensor_msgs::Imu> imuQueOpt;
    //imuQueImu用来给imuIntegratorImu_提供数据来源，不要的就弹出(弹出当前激光里程计之前的imu数据,预积分用完一个弹一个)； 
    std::deque<sensor_msgs::Imu> imuQueImu;

    // imu 因子图优化过程中的状态变量
    gtsam::Pose3 prevPose_;//六自由度pose
    gtsam::Vector3 prevVel_;//速度
    gtsam::NavState prevState_;
    gtsam::imuBias::ConstantBias prevBias_;
    // imu 状态中转存储
    gtsam::NavState prevStateOdom;
    gtsam::imuBias::ConstantBias prevBiasOdom;
    
    // ISAM2优化器
    bool doneFirstOpt = false;
    double lastImuT_imu = -1;
    double lastImuT_opt = -1;
    //优化器贝叶斯树加速 
    gtsam::ISAM2 optimizer;
    gtsam::NonlinearFactorGraph graphFactors;//总的因子图模型,实例化一个因子图
    gtsam::Values graphValues;//因子图模型中的值，计算

    const double delta_t = 0;

    int key = 1;
    int imuPreintegrationResetId = 0;

    // imu-lidar位姿变换
    //这点要注意，这只是一个平移变换，
    //同样头文件的imuConverter中，也只有一个旋转变换。这里绝对不可以理解为把imu数据转到lidar下的变换矩阵。
    //事实上，作者后续是把imu数据先用imuConverter旋转到雷达系下（但其实还差了个平移）。
    //作者真正是把雷达数据又根据lidar2Imu反向平移了一下，和转换以后差了个平移的imu数据在“中间系”对齐，
    //之后算完又从中间系通过imu2Lidar挪回了雷达系进行publish。
    gtsam::Pose3 imu2Lidar = gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0), gtsam::Point3(-extTrans.x(), -extTrans.y(), -extTrans.z()));// gtsam::Pose3六自由度
    gtsam::Pose3 lidar2Imu = gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0), gtsam::Point3(extTrans.x(), extTrans.y(), extTrans.z()));;


    IMUPreintegration()
    {   
        //接收imu话题，// imuTopic 为 imu_correct,并调用imuHandler
        subImu      = nh.subscribe<sensor_msgs::Imu>  (imuTopic, 2000, &IMUPreintegration::imuHandler, this, ros::TransportHints().tcpNoDelay());
        //两帧之间的imu加上地图匹配优化得出的位姿   做因子图优化
        subOdometry = nh.subscribe<nav_msgs::Odometry>(PROJECT_NAME + "/lidar/mapping/odometry", 5, &IMUPreintegration::odometryHandler, this, ros::TransportHints().tcpNoDelay());

        pubImuOdometry = nh.advertise<nav_msgs::Odometry> ("odometry/imu", 2000);
        pubImuPath     = nh.advertise<nav_msgs::Path>     (PROJECT_NAME + "/lidar/imu/path", 1);

        map_to_odom = tf::Transform(tf::createQuaternionFromRPY(0, 0, 0), tf::Vector3(0, 0, 0));
        
       
       
       
       
       
       
        //用于imu预积分的一些变量 这块是gtsam涉及的比较多
        boost::shared_ptr<gtsam::PreintegrationParams> p = gtsam::PreintegrationParams::MakeSharedU(imuGravity);//gtsam::PreintegrationParams参数设置接口
        p->accelerometerCovariance  = gtsam::Matrix33::Identity(3,3) * pow(imuAccNoise, 2); // 加速度计噪音
        p->gyroscopeCovariance      = gtsam::Matrix33::Identity(3,3) * pow(imuGyrNoise, 2); // 陀螺仪噪音
        p->integrationCovariance    = gtsam::Matrix33::Identity(3,3) * pow(1e-4, 2); // error committed in integrating position from velocities速度积分位置时出现的错误
        gtsam::imuBias::ConstantBias prior_imu_bias((gtsam::Vector(6) << 0, 0, 0, 0, 0, 0).finished());; // 估计零偏

        //噪声先验，初值没什么道理
        priorPoseNoise  = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2).finished()); // rad,rad,rad,m, m, m
        priorVelNoise   = gtsam::noiseModel::Isotropic::Sigma(3, 1e2); // m/s
        priorBiasNoise  = gtsam::noiseModel::Isotropic::Sigma(6, 1e-3); // 1e-2 ~ 1e-3 seems to be good
        //退化保险，如果发生退化怎选择更大的协方差
        correctionNoise = gtsam::noiseModel::Isotropic::Sigma(6, 1e-2); // meter
        noiseModelBetweenBias = (gtsam::Vector(6) << imuAccBiasN, imuAccBiasN, imuAccBiasN, imuGyrBiasN, imuGyrBiasN, imuGyrBiasN).finished();
        //imu预积分器，用于预测每一时刻的（imu频率）的imu里程计，与激光统一坐标系
        imuIntegratorImu_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias); // setting up the IMU integration for IMU message thread
        //imu与积分器，用于图优化
        imuIntegratorOpt_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias); // setting up the IMU integration for optimization        
    }

    // gtsam相关优化参数重置与初始化
    void resetOptimization()
    {
        gtsam::ISAM2Params optParameters;
        optParameters.relinearizeThreshold = 0.1;
        optParameters.relinearizeSkip = 1;
        optimizer = gtsam::ISAM2(optParameters);

        gtsam::NonlinearFactorGraph newGraphFactors;
        graphFactors = newGraphFactors;

        gtsam::Values NewGraphValues;
        graphValues = NewGraphValues;
    }

    // 对这几个变量进行重置
    void resetParams()
    {
        lastImuT_imu = -1;
        doneFirstOpt = false;
        systemInitialized = false;
    }
    
    //激光里程计话题回调
    //1.每间隔100帧激光里程计，重置isam2优化器，添加里程计、速度、偏置先验因子、执行优化
    //2.计算前一帧激光里程计和当前帧激光里程计之间的imu预计分量，用前一阵转台施加预积分量得到当前帧初始状态估计，嘉盛map的当前帧位姿进行图优化，更新当前帧状态
    //3.优化后进行重传播，优化更新imu偏置，用最新偏置重新计算当前激光里程计时刻之后的imu预积分，预积分用于计算美食客位姿
    void odometryHandler(const nav_msgs::Odometry::ConstPtr& odomMsg)
    {
        //里程计消息的当前时间戳
        double currentCorrectionTime = ROS_TIME(odomMsg);
        // make sure we have imu data to integrate
        //保证有imu数据  两个回调函数是互有联系的  在imu的回调里就强调要完成一次优化才往下执行
        if (imuQueOpt.empty())
            return;
         
        //通过里程计话题获得位置信息 四元数
        float p_x = odomMsg->pose.pose.position.x;
        float p_y = odomMsg->pose.pose.position.y;
        float p_z = odomMsg->pose.pose.position.z;
        float r_x = odomMsg->pose.pose.orientation.x;
        float r_y = odomMsg->pose.pose.orientation.y;
        float r_z = odomMsg->pose.pose.orientation.z;
        float r_w = odomMsg->pose.pose.orientation.w;
        int currentResetId = round(odomMsg->pose.covariance[0]);
        gtsam::Pose3 lidarPose = gtsam::Pose3(gtsam::Rot3::Quaternion(r_w, r_x, r_y, r_z), gtsam::Point3(p_x, p_y, p_z));//转换成gtsam格式下的位姿

        // correction pose jumped, reset imu pre-integration
        //得到雷达的位姿 后续用到 比较关键的一个量
        if (currentResetId != imuPreintegrationResetId)//参数控制，协方差要一样
        {
            resetParams();
            imuPreintegrationResetId = currentResetId;
            return;
        }


        // 0. initialize system// 0. 系统初始化，第一帧
        if (systemInitialized == false)
        {
            resetOptimization();//定义优化器

            // pop old IMU message// 重置ISAM2优化器
            while (!imuQueOpt.empty())
            {
                if (ROS_TIME(&imuQueOpt.front()) < currentCorrectionTime - delta_t)//只需要当前激光帧之后的imu
                {
                    lastImuT_opt = ROS_TIME(&imuQueOpt.front());//记录时间
                    imuQueOpt.pop_front();
                }
                else
                    break;
            }
            
            // initial pose // 用scantomap的值添加里程计位姿先验因子
            prevPose_ = lidarPose.compose(lidar2Imu);//转换到imu坐标系下的里程计位姿
            gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, priorPoseNoise);//X(0)第一个位姿，prevPose_先验约束
            graphFactors.add(priorPose);
            // initial velocity  // 添加速度先验因子
            //虽然设置为0，但是作者不信任，则让他置信度很小，协方差很大
            prevVel_ = gtsam::Vector3(0, 0, 0);
            gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, priorVelNoise);
            graphFactors.add(priorVel);
            // initial bias 添加偏差因子
            prevBias_ = gtsam::imuBias::ConstantBias();
            gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, priorBiasNoise);
            graphFactors.add(priorBias);
            
            
            
            
            // add values// 变量节点赋初值
            graphValues.insert(X(0), prevPose_);
            graphValues.insert(V(0), prevVel_);
            graphValues.insert(B(0), prevBias_);
            // optimize once // 优化一次
            optimizer.update(graphFactors, graphValues);
            graphFactors.resize(0);
            graphValues.clear();
            // 重置优化之后的偏置
            imuIntegratorImu_->resetIntegrationAndSetBias(prevBias_);
            imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);
            
            key = 1;
            systemInitialized = true;
            return;
        }

        // reset graph for speed // 每隔100帧激光里程计，重置ISAM2优化器，保证优化效率
        if (key == 100)
        {
            // get updated noise before reset// 前一帧的位姿、速度、偏置噪声模型
            gtsam::noiseModel::Gaussian::shared_ptr updatedPoseNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(X(key-1)));
            gtsam::noiseModel::Gaussian::shared_ptr updatedVelNoise  = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(V(key-1)));
            gtsam::noiseModel::Gaussian::shared_ptr updatedBiasNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(B(key-1)));
            // reset graph// 重置ISAM2优化器
            resetOptimization();
            // add pose // 添加位姿先验因子，用前一帧的值初始化
            gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, updatedPoseNoise);
            graphFactors.add(priorPose);
            // add velocity // 添加速度先验因子，用前一帧的值初始化
            gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, updatedVelNoise);
            graphFactors.add(priorVel);
            // add bias // 添加偏置先验因子，用前一帧的值初始化
            gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, updatedBiasNoise);
            graphFactors.add(priorBias);
            // add values // 变量节点赋初值，用前一帧的值初始化
            graphValues.insert(X(0), prevPose_);
            graphValues.insert(V(0), prevVel_);
            graphValues.insert(B(0), prevBias_);
            // optimize once// 优化一次
            optimizer.update(graphFactors, graphValues);
            graphFactors.resize(0);
            graphValues.clear();

            key = 1;
        }


        // 1. integrate imu data and optimize 1. 计算前一帧与当前帧之间的imu预积分量，用前一帧状态施加预积分量得到当前帧初始状态估计
        while (!imuQueOpt.empty())// 添加来自mapOptimization的当前帧位姿，进行因子图优化，更新当前帧状态
        {
            // pop and integrate imu data that is between two optimizations
            sensor_msgs::Imu *thisImu = &imuQueOpt.front();
            double imuTime = ROS_TIME(thisImu);
            if (imuTime < currentCorrectionTime - delta_t)
            {

                
                double dt = (lastImuT_opt < 0) ? (1.0 / 500.0) : (imuTime - lastImuT_opt);
                //输入参数到预积分器
                imuIntegratorOpt_->integrateMeasurement(
                        gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
                        gtsam::Vector3(thisImu->angular_velocity.x,    thisImu->angular_velocity.y,    thisImu->angular_velocity.z), dt);
                
                lastImuT_opt = imuTime;
                imuQueOpt.pop_front();
            }
            else
                break;
        }
        // add imu factor to graph
        //添加预积分因子
        const gtsam::PreintegratedImuMeasurements& preint_imu = dynamic_cast<const gtsam::PreintegratedImuMeasurements&>(*imuIntegratorOpt_);
        //参数：前一帧位姿，前一帧速度，当前帧位姿，当前帧速度，前一帧bias，预积分量
        gtsam::ImuFactor imu_factor(X(key - 1), V(key - 1), X(key), V(key), B(key - 1), preint_imu);
        graphFactors.add(imu_factor);
        // add imu bias between factorc  
        // 添加imubias因子,前一帧偏值，当前帧偏值，观测，噪声协方差，deltaTij()积分时间段
        graphFactors.add(gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(B(key - 1), B(key), gtsam::imuBias::ConstantBias(),
                         gtsam::noiseModel::Diagonal::Sigmas(sqrt(imuIntegratorOpt_->deltaTij()) * noiseModelBetweenBias)));
        // add pose factor  位姿因子
        gtsam::Pose3 curPose = lidarPose.compose(lidar2Imu);
        //退化标志在LIO中  gtsam::PriorFactor<gtsam::Pose3> pose_factor(X(key), curPose,degenerate ? correctionNoise2 : correctionNoise);
        gtsam::PriorFactor<gtsam::Pose3> pose_factor(X(key), curPose, correctionNoise);
        graphFactors.add(pose_factor);
        // insert predicted values
        gtsam::NavState propState_ = imuIntegratorOpt_->predict(prevState_, prevBias_);
        graphValues.insert(X(key), propState_.pose());
        graphValues.insert(V(key), propState_.v());
        graphValues.insert(B(key), prevBias_);
        
        // optimize//优化两次
        optimizer.update(graphFactors, graphValues);
        optimizer.update();
        graphFactors.resize(0);
        graphValues.clear();
        // Overwrite the beginning of the preintegration for the next step.
        // 为下一步覆盖预积分的开始。更新
        gtsam::Values result = optimizer.calculateEstimate();
        prevPose_  = result.at<gtsam::Pose3>(X(key));
        prevVel_   = result.at<gtsam::Vector3>(V(key));
        prevState_ = gtsam::NavState(prevPose_, prevVel_);
        prevBias_  = result.at<gtsam::imuBias::ConstantBias>(B(key));
        // Reset the optimization preintegration object.  预积分器重置，设置新的bias
        imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);
        // check optimization失败检测，检查imu优化结果，如果速度或者bias过大则认为优化失败
        if (failureDetection(prevVel_, prevBias_))
        {
            resetParams();//重置
            return;
        }


        // 2. after optiization, re-propagate imu odometry preintegration
        // 2. 优化后，重新传播imu预积分量；优化更新了imu的偏置，用最新偏置从新计算当前激光里程计时刻之后的imu预积分量，这个预积分由于计算位姿
        prevStateOdom = prevState_;
        prevBiasOdom  = prevBias_;
        // first pop imu message older than current correction data
        //删除当前里程计时刻之前的imu数据
        double lastImuQT = -1;
        while (!imuQueImu.empty() && ROS_TIME(&imuQueImu.front()) < currentCorrectionTime - delta_t)
        {
            lastImuQT = ROS_TIME(&imuQueImu.front());
            imuQueImu.pop_front();
        }
        // repropogate对剩余的imu计算预积分量
        if (!imuQueImu.empty())
        {
            // reset bias use the newly optimized bias用最新bias重置预积分器
            imuIntegratorImu_->resetIntegrationAndSetBias(prevBiasOdom);
            // integrate imu message from the beginning of this optimization
            for (int i = 0; i < (int)imuQueImu.size(); ++i)//计算预积分
            {
                sensor_msgs::Imu *thisImu = &imuQueImu[i];
                double imuTime = ROS_TIME(thisImu);
                double dt = (lastImuQT < 0) ? (1.0 / 500.0) :(imuTime - lastImuQT);

                imuIntegratorImu_->integrateMeasurement(gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
                                                        gtsam::Vector3(thisImu->angular_velocity.x,    thisImu->angular_velocity.y,    thisImu->angular_velocity.z), dt);
                lastImuQT = imuTime;
            }
        }

        ++key;
        doneFirstOpt = true;
    }
    //失败检测
    bool failureDetection(const gtsam::Vector3& velCur, const gtsam::imuBias::ConstantBias& biasCur)
    {
        Eigen::Vector3f vel(velCur.x(), velCur.y(), velCur.z());
        if (vel.norm() > 30)//速度过大
        {
            ROS_WARN("Large velocity, reset IMU-preintegration!");
            return true;
        }

        Eigen::Vector3f ba(biasCur.accelerometer().x(), biasCur.accelerometer().y(), biasCur.accelerometer().z());
        Eigen::Vector3f bg(biasCur.gyroscope().x(), biasCur.gyroscope().y(), biasCur.gyroscope().z());
        if (ba.norm() > 0.1 || bg.norm() > 0.1)//bias过大
        {
            ROS_WARN("Large bias, reset IMU-preintegration!");
            return true;
        }

        return false;
    }

    // 使用 gtsam 对imu进行预积分
    void imuHandler(const sensor_msgs::Imu::ConstPtr& imuMsg)
    {   
        //坐标系转换，imu数据转换到雷达坐标系下， 调用utility.h中函数imuConverter
        sensor_msgs::Imu thisImu = imuConverter(*imuMsg);
        // publish static tf
        // tf::StampedTransform
        // 发布tf变换树的类型，一个变换树包含 变换、时间戳、父坐标系frame_id、子坐标系frame_id；
        // tf::StampedTransform(transform, ros::Time::now(), “turtle1”, “carrot1”)
        tfMap2Odom.sendTransform(tf::StampedTransform(map_to_odom, thisImu.header.stamp, "map", "odom"));
        // 两个双端队列分别装着优化前后的imu数据
        imuQueOpt.push_back(thisImu);
        imuQueImu.push_back(thisImu);



        //没优化就会在这卡主
        //检查有没有执行过一次优化  也就是说这里需要先在odomhandler中优化一次后再进行该函数后续的工作
        if (doneFirstOpt == false)
            return;
        //获得时间间隔 第一次为1/500 之后是两次imuTime间的差
        double imuTime = ROS_TIME(&thisImu);
        double dt = (lastImuT_imu < 0) ? (1.0 / 500.0) : (imuTime - lastImuT_imu);
        lastImuT_imu = imuTime;

        // integrate this single imu message //进行预积分
        imuIntegratorImu_->integrateMeasurement(gtsam::Vector3(thisImu.linear_acceleration.x, thisImu.linear_acceleration.y, thisImu.linear_acceleration.z),
                                                gtsam::Vector3(thisImu.angular_velocity.x,    thisImu.angular_velocity.y,    thisImu.angular_velocity.z), dt);

        // predict odometry//根据预积分预测odom值，得到当前状态
        gtsam::NavState currentState = imuIntegratorImu_->predict(prevStateOdom, prevBiasOdom);

        // publish odometry //8.旋转后的imu再通过平移转换到雷达坐标系，存入odometry并发布
        nav_msgs::Odometry odometry;
        odometry.header.stamp = thisImu.header.stamp;
        odometry.header.frame_id = "odom";
        odometry.child_frame_id = "odom_imu";

        // transform imu pose to ldiar//预测值currentState获得imu位姿 再由imu到雷达变换 获得雷达位姿
        gtsam::Pose3 imuPose = gtsam::Pose3(currentState.quaternion(), currentState.position());
        //本程序文件开头定义了imu2Lidar  与params.yaml中外参矩阵有关
        gtsam::Pose3 lidarPose = imuPose.compose(imu2Lidar);
        //IMU里程计的相关数据填充 
        odometry.pose.pose.position.x = lidarPose.translation().x();
        odometry.pose.pose.position.y = lidarPose.translation().y();
        odometry.pose.pose.position.z = lidarPose.translation().z();
        odometry.pose.pose.orientation.x = lidarPose.rotation().toQuaternion().x();
        odometry.pose.pose.orientation.y = lidarPose.rotation().toQuaternion().y();
        odometry.pose.pose.orientation.z = lidarPose.rotation().toQuaternion().z();
        odometry.pose.pose.orientation.w = lidarPose.rotation().toQuaternion().w();
        
        odometry.twist.twist.linear.x = currentState.velocity().x();
        odometry.twist.twist.linear.y = currentState.velocity().y();
        odometry.twist.twist.linear.z = currentState.velocity().z();
        odometry.twist.twist.angular.x = thisImu.angular_velocity.x + prevBiasOdom.gyroscope().x();
        odometry.twist.twist.angular.y = thisImu.angular_velocity.y + prevBiasOdom.gyroscope().y();
        odometry.twist.twist.angular.z = thisImu.angular_velocity.z + prevBiasOdom.gyroscope().z();
        // information for VINS initialization
        odometry.pose.covariance[0] = double(imuPreintegrationResetId);
        odometry.pose.covariance[1] = prevBiasOdom.accelerometer().x();
        odometry.pose.covariance[2] = prevBiasOdom.accelerometer().y();
        odometry.pose.covariance[3] = prevBiasOdom.accelerometer().z();
        odometry.pose.covariance[4] = prevBiasOdom.gyroscope().x();
        odometry.pose.covariance[5] = prevBiasOdom.gyroscope().y();
        odometry.pose.covariance[6] = prevBiasOdom.gyroscope().z();
        odometry.pose.covariance[7] = imuGravity;
        pubImuOdometry.publish(odometry);

        // publish imu path
        static nav_msgs::Path imuPath;
        static double last_path_time = -1;
        if (imuTime - last_path_time > 0.1)
        {
            last_path_time = imuTime;
            geometry_msgs::PoseStamped pose_stamped;
            pose_stamped.header.stamp = thisImu.header.stamp;
            pose_stamped.header.frame_id = "odom";
            pose_stamped.pose = odometry.pose.pose;
            imuPath.poses.push_back(pose_stamped);
            while(!imuPath.poses.empty() && abs(imuPath.poses.front().header.stamp.toSec() - imuPath.poses.back().header.stamp.toSec()) > 3.0)
                imuPath.poses.erase(imuPath.poses.begin());
            if (pubImuPath.getNumSubscribers() != 0)
            {
                imuPath.header.stamp = thisImu.header.stamp;
                imuPath.header.frame_id = "odom";
                pubImuPath.publish(imuPath);
            }
        }

        // publish transformation
        tf::Transform tCur;
        tf::poseMsgToTF(odometry.pose.pose, tCur);
        tf::StampedTransform odom_2_baselink = tf::StampedTransform(tCur, thisImu.header.stamp, "odom", "base_link");
        tfOdom2BaseLink.sendTransform(odom_2_baselink);
    }
};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "lidar");
    
    IMUPreintegration ImuP;

    ROS_INFO("\033[1;32m----> Lidar IMU Preintegration Started.\033[0m");

    ros::spin();
    
    return 0;
}