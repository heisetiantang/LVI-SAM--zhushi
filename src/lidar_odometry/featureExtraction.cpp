#include "utility.h"
#include "lvi_sam/cloud_info.h"

struct smoothness_t{ 
    float value;
    size_t ind;
};

struct by_value{ 
    bool operator()(smoothness_t const &left, smoothness_t const &right) { 
        return left.value < right.value;
    }
};

class FeatureExtraction : public ParamServer
{

public:
    
    ros::Subscriber subLaserCloudInfo;
    
    
    
    // 发布当前激光帧提取特征之后的点云信息
    ros::Publisher pubLaserCloudInfo;
     // 发布当前激光帧提取的角点点云
    ros::Publisher pubCornerPoints;//角点
     // 发布当前激光帧提取的平面点点云
    ros::Publisher pubSurfacePoints;//面点

    pcl::PointCloud<PointType>::Ptr extractedCloud;// 保存有效点
    pcl::PointCloud<PointType>::Ptr cornerCloud;// 保存角点
    pcl::PointCloud<PointType>::Ptr surfaceCloud; // 保存面点
    pcl::PointCloud<PointType> extractedCloudDS; // 保存面点的点云采样点云
   //创建//创建滤波器对象
    pcl::StatisticalOutlierRemoval<PointType> sor;

    //****************设置一个值来保存比例值
    //vector<float> ratio_value;
    int rotioofedge2surf =0.1;
    float  edgeNum = 0;
    float  surfaceNum = 0;
    float  rotiofactor = 1;//比例系数
    float  rotiofactorsurf = 1;




    
    //创建体素downSizeFilter
    pcl::VoxelGrid<PointType> downSizeFilter;

    lvi_sam::cloud_info cloudInfo;
    std_msgs::Header cloudHeader;// 发布topic时的时间戳

    std::vector<smoothness_t> cloudSmoothness;// 存储每个点的曲率与索引
    float *cloudCurvature;//用来做曲率计算的中间变量
    int *cloudNeighborPicked;// 特征提取标记，1表示遮挡、平行，或者已经进行特征提取的点，0表示还未进行特征提取处理
    int *cloudLabel;// 标记面点的索引// 1表示角点，-1表示平面点

    FeatureExtraction()
    {
        subLaserCloudInfo = nh.subscribe<lvi_sam::cloud_info>(PROJECT_NAME + "/lidar/deskew/cloud_info", 5, &FeatureExtraction::laserCloudInfoHandler, this, ros::TransportHints().tcpNoDelay());
        
        

        pubLaserCloudInfo = nh.advertise<lvi_sam::cloud_info> (PROJECT_NAME + "/lidar/feature/cloud_info", 5);
        pubCornerPoints = nh.advertise<sensor_msgs::PointCloud2>(PROJECT_NAME + "/lidar/feature/cloud_corner", 5);
        pubSurfacePoints = nh.advertise<sensor_msgs::PointCloud2>(PROJECT_NAME + "/lidar/feature/cloud_surface", 5);
        
        initializationValue();
    }
    //1.:降采样及复原
    void initializationValue()
    {
        cloudSmoothness.resize(N_SCAN*Horizon_SCAN);
        // 降采样的参数0.4
        downSizeFilter.setLeafSize(odometrySurfLeafSize, odometrySurfLeafSize, odometrySurfLeafSize);//调整大小

        extractedCloud.reset(new pcl::PointCloud<PointType>());//复原
        cornerCloud.reset(new pcl::PointCloud<PointType>());//复原
        surfaceCloud.reset(new pcl::PointCloud<PointType>());//复原

        cloudCurvature = new float[N_SCAN*Horizon_SCAN];//复原
        cloudNeighborPicked = new int[N_SCAN*Horizon_SCAN];//复原
        cloudLabel = new int[N_SCAN*Horizon_SCAN];//复原
    }
    //0.:回调函数 //接收imageProjection.cpp中发布的去畸变的点云，实时处理的回调函数
    void laserCloudInfoHandler(const lvi_sam::cloud_infoConstPtr& msgIn)
    {
        cloudInfo = *msgIn; // new cloud info 输入新信息//msgIn即为回调函数获取的去畸变点云信息
        cloudHeader = msgIn->header; // new cloud header 输入新信息地头
        pcl::fromROSMsg(msgIn->cloud_deskewed, extractedCloudDS); // new cloud for extraction
         
        //去除点云的离群点
        sor.setInputCloud (extractedCloudDS); //设置待滤波的点云
        sor.setMeanK (50);                               //设置在进行统计时考虑的临近点个数
        sor.setStddevMulThresh (1.0);                      //设置判断是否为离群点的阀值，用来倍乘标准差，也就是上面的std_mul
        sor.filter (*extractedCloud);                    //滤波结果存储到cloud_filtered



        //0.0.平滑度计算，目的是为了区分出边缘点和平面点（仅仅记录了值，并没有区分函数） 
        calculateSmoothness();
        //0.1.标记出被遮挡的点 参考LOAM论文的介绍
        markOccludedPoints();
        //0.2.提取特征
        //将每一层点云分成6份，每一份中，对点的曲率进行排序，sp和ep分别是这段点云的起始点与终止点。 从而判断出角点与平面点，
        extractFeatures();
        //0.3.发布特征点云
        publishFeatureCloud();
    }

    //0.0.平滑度计算  为了区分出边缘点和平面点 
    void calculateSmoothness()
    {
        int cloudSize = extractedCloud->points.size();
        //这里跟loam源码基本很像 从第开始的五个点开始到结束前第5个点， 计算某个点的平滑度
        for (int i = 5; i < cloudSize - 5; i++)
        {
            //前五个点的距离属性（在imageProjection.cpp中被赋值）之和加后五个点的距离之和-10倍该点的距离 算出差值
            //其实就是确定一种连续点之间的变化关系 起伏趋势
            float diffRange = cloudInfo.pointRange[i-5] + cloudInfo.pointRange[i-4]
                            + cloudInfo.pointRange[i-3] + cloudInfo.pointRange[i-2]
                            + cloudInfo.pointRange[i-1] - cloudInfo.pointRange[i] * 10
                            + cloudInfo.pointRange[i+1] + cloudInfo.pointRange[i+2]
                            + cloudInfo.pointRange[i+3] + cloudInfo.pointRange[i+4]
                            + cloudInfo.pointRange[i+5];            
            //点云曲率 
            cloudCurvature[i] = diffRange*diffRange;//diffX * diffX + diffY * diffY + diffZ * diffZ;
            //是否被选择值置0
            cloudNeighborPicked[i] = 0;
            //标签值置0
            cloudLabel[i] = 0;
            // cloudSmoothness for sorting //记录曲率的值和索引
            cloudSmoothness[i].value = cloudCurvature[i];
            cloudSmoothness[i].ind = i;
        }
    }

    //0.1.标记出被遮挡的点 参考LOAM论文的介绍
    //主要是完成了该函数后cloudNeighborPicked中有了点是否选择为特征点的标记
    void markOccludedPoints()
    {
        int cloudSize = extractedCloud->points.size();
        // mark occluded points and parallel beam points
        //标记遮挡点和平行光束点 看到这个就知道完全与LOAM的思想一致 
        for (int i = 5; i < cloudSize - 6; ++i)
        {
            // occluded points// 当前点和下一个点的range值
            //两个点的深度也就是雷达到障碍物的距离
            float depth1 = cloudInfo.pointRange[i];
            float depth2 = cloudInfo.pointRange[i+1];
            //距离图像中列上下两值得的差值
            int columnDiff = std::abs(int(cloudInfo.pointColInd[i+1] - cloudInfo.pointColInd[i]));//绝对值的差值
            
            //根据深度差异 进行区分 并设定标志变量为1
            if (columnDiff < 10)
            {
                // 10 pixel diff in range image
                if (depth1 - depth2 > 0.3)
                {
                    cloudNeighborPicked[i - 5] = 1;
                    cloudNeighborPicked[i - 4] = 1;
                    cloudNeighborPicked[i - 3] = 1;
                    cloudNeighborPicked[i - 2] = 1;
                    cloudNeighborPicked[i - 1] = 1;
                    cloudNeighborPicked[i] = 1;
                }
                else if (depth2 - depth1 > 0.3)
                {
                    cloudNeighborPicked[i + 1] = 1;
                    cloudNeighborPicked[i + 2] = 1;
                    cloudNeighborPicked[i + 3] = 1;
                    cloudNeighborPicked[i + 4] = 1;
                    cloudNeighborPicked[i + 5] = 1;
                    cloudNeighborPicked[i + 6] = 1;
                }
            }
            // parallel beam
            // parallel beam 平行线的情况 根据左右两点与该点的深度差 确定该点是否会被选择为特征点
            float diff1 = std::abs(float(cloudInfo.pointRange[i-1] - cloudInfo.pointRange[i]));
            float diff2 = std::abs(float(cloudInfo.pointRange[i+1] - cloudInfo.pointRange[i]));

            if (diff1 > 0.02 * cloudInfo.pointRange[i] && diff2 > 0.02 * cloudInfo.pointRange[i])
                cloudNeighborPicked[i] = 1;
        }
    }

    //0.2提取特征
    //将每一层点云分成6份，每一份中，对点的曲率进行排序，sp和ep分别是这段点云的起始点与终止点。 从而判断出角点与平面点，
    //保存角点到cornerCloud
    // 面点临时保存在surfaceCloudScan
    void extractFeatures()
    {
        //角点和平面点 点云清空下面准备存进来对应点云了
        cornerCloud->clear();
        surfaceCloud->clear();

        pcl::PointCloud<PointType>::Ptr surfaceCloudScan(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surfaceCloudScanDS(new pcl::PointCloud<PointType>());
        //16 32 64 线数，判断
        for (int i = 0; i < N_SCAN; i++)
        {
              //********************调用上一帧的角点和平面点的比例，赋值给阀值edgeThreshold 
            //  if (i > 1)
               
            //     edgeThreshold =  edgeThreshold + rotiofactor * edgeThreshold * ((ratio_value[i - 1] - ratio_value[ i - 2 ] )/ratio_value[ i - 1] );
            //     surfThreshold = surfThreshold + rotiofactorsurf * surfThreshold * ((ratio_value[i - 1] - ratio_value[i - 2 ] )/ratio_value[ i - 1] );



            surfaceCloudScan->clear();
            // LIOSAM里只分了边缘点与平面点 没有对地面点进行特定的区分
            //这里是每根线分成6块处理的操作
            for (int j = 0; j < 6; j++)
            {
                // 第一份的索引就是 startRingIndex* (6-j)/6 + endRingInde * j/6   
                int sp = (cloudInfo.startRingIndex[i] * (6 - j) + cloudInfo.endRingIndex[i] * j) / 6;
                // ep 就是sp的下一个循环的值的前一个索引，ep[j] = sp[j+1] - 1
                int ep = (cloudInfo.startRingIndex[i] * (5 - j) + cloudInfo.endRingIndex[i] * (j + 1)) / 6 - 1;

                if (sp >= ep)
                    continue;
                //按照曲率从小到大排序，排序
                std::sort(cloudSmoothness.begin()+sp, cloudSmoothness.begin()+ep, by_value());
                //边缘点选取不在平面上 
                int largestPickedNum = 0;
                
                // 最后的点的曲率最大，如果满足条件，就是角点
                for (int k = ep; k >= sp; k--)
                {
                    int ind = cloudSmoothness[k].ind;
                    if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] > edgeThreshold)// edgeThreshold边缘阈值为0.1，正圆的曲率为0
                    {
                        largestPickedNum++;
                        if (largestPickedNum <= 20)
                        {
                            cloudLabel[ind] = 1;// 都是角点了肯定不是面点
                            cornerCloud->push_back(extractedCloud->points[ind]);//保存角点到cornerCloud
                        } 
                        else
                        {
                            break;
                        }
                        cloudNeighborPicked[ind] = 1;

                        // 防止特征点聚集，将ind及其前后各5个点标记，不做特征点提取
                        for (int l = 1; l <= 5; l++)
                        {
                            // 每个点index之间的差值。附近点都是有效点的情况下，相邻点间的索引只差１
                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l - 1]));
                            // 附近有无效点，或者是每条线的起点和终点的部分
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--)
                        {
                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l + 1]));
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }

                // 进行面点的提取,同理从前往后遍历，曲率最小的是面点
                for (int k = sp; k <= ep; k++)
                {
                    int ind = cloudSmoothness[k].ind;
                    if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] < surfThreshold)
                    {
                        // 标记面点的索引的值为-1   
                        cloudLabel[ind] = -1;
                        cloudNeighborPicked[ind] = 1;
                        // 这个点及前后各5个点不再进行提取特征，防止平面点聚集
                        for (int l = 1; l <= 5; l++) 、
                        {
                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l - 1]));
                            // 附近有无效点，或者是每条线的起点和终点的部分
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--) {

                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l + 1]));
                            if (columnDiff > 10)
                                break;

                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }
                
                // 面点临时保存在surfaceCloudScan
                for (int k = sp; k <= ep; k++)
                {
                    if (cloudLabel[k] <= 0){
                        surfaceCloudScan->push_back(extractedCloud->points[k]);
                    }
                }
            }

            // //保存角点数目到edgeCloudScan**********************
            // //保存平面点数目到surfaceCloudScan
            // edgeNum = sizeof(cornerCloud);
            // surfaceNum = sizeof(surfaceCloudScan);
            // //求角点和平面点的比例，并输出
            // rotioofedge2surf = edgeNum / surfaceNum;
            // cout << "rotioofedge2surf: " << rotioofedge2surf << endl;
            // //将比例值保存在容器中
            // ratio_value.push_back(rotioofedge2surf);
            
            // //**********************

            
             ->clear();
            downSizeFilter.setInputCloud(surfaceCloudScan);
            downSizeFilter.filter(*surfaceCloudScanDS);

            *surfaceCloud += *surfaceCloudScanDS;





            // 提取上一帧的角点和平面点的比例



        }
    }

    //0.3.1清理
    void freeCloudInfoMemory()
    {
        cloudInfo.startRingIndex.clear();//清零
        cloudInfo.startRingIndex.shrink_to_fit();//减少容器的容量以适应其大小
        cloudInfo.endRingIndex.clear();
        cloudInfo.endRingIndex.shrink_to_fit();
        cloudInfo.pointColInd.clear();
        cloudInfo.pointColInd.shrink_to_fit();
        cloudInfo.pointRange.clear();
        cloudInfo.pointRange.shrink_to_fit();
    }
    //0.3.发布特征点云
    void publishFeatureCloud()
    {
        // free cloud info memory
        freeCloudInfoMemory();
        // save newly extracted features//保存新的特征提取
        cloudInfo.cloud_corner  = publishCloud(&pubCornerPoints,  cornerCloud,  cloudHeader.stamp, "base_link");
        cloudInfo.cloud_surface = publishCloud(&pubSurfacePoints, surfaceCloud, cloudHeader.stamp, "base_link");
        // publish to mapOptimization
        pubLaserCloudInfo.publish(cloudInfo);
    }
};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "lidar");

    FeatureExtraction FE;

    ROS_INFO("\033[1;32m----> Lidar Feature Extraction Started.\033[0m");
   
    ros::spin();

    return 0;
}