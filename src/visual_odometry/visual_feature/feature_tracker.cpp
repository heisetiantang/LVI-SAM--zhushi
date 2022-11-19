#include "feature_tracker.h"

int FeatureTracker::n_id = 0;

// 判断特征点是否在图像范围内；
bool inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE;
}

// 根据跟踪状态向量去除没有跟踪到的特征点；
void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}


FeatureTracker::FeatureTracker()
{
}

//设置角点检测函数的mask
//1.对跟踪到的特征点对依据跟踪次数进行从多到少的排序，并放到原集合中
void FeatureTracker::setMask()
{
    if(FISHEYE)
        mask = fisheye_mask.clone();
    else
        mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));
    

    // prefer to keep features that are tracked for long time
        //保存长时间跟踪到的特征点
        //vector<pair<某一点跟踪次数，pair<某一点，某一点的id>>> cnt_pts_id
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;
    for (unsigned int i = 0; i < forw_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(forw_pts[i], ids[i])));

   //对给定区间的所有元素进行排序，按照点的跟踪次数，从多到少进行排序
    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
         {
            //降序排列
            return a.first > b.first;
         });

    forw_pts.clear();
    ids.clear();
    track_cnt.clear();

    for (auto &it : cnt_pts_id)
    {
        // 只保留指定掩模区域内的特征点//检测新建的mask在该点是否为255
        if (mask.at<uchar>(it.second.first) == 255)
        {
            //将跟踪到的点按照跟踪次数重新排列，并返回到forw_pts，ids，track_cnt
            forw_pts.push_back(it.second.first);
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first);
            // 将已有特征点的区域灰度置零，以该点为中心，最小距离为半径的实心圆；该区域就不用再检测特征点了；提供给特征检测函数用；
            //图片，点，半径，颜色为0表示在角点检测在该点不起作用,粗细（-1）表示填充
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
        }
    }
}

void FeatureTracker::addPoints()
{
    for (auto &p : n_pts)
    {
        forw_pts.push_back(p);
        ids.push_back(-1);
        track_cnt.push_back(1);
    }
}

//去读取图像，计算提取特征点，跟踪特征点；
void FeatureTracker::readImage(const cv::Mat &_img, double _cur_time)
{
    cv::Mat img;
    TicToc t_r;//计时器；
    //添将新检测到的特征点n_pts添加到forw_pts中，id初始化-1,
    //track_cnt初始化为1.
    cur_time = _cur_time;

    //1、针对暗淡图像进行直方图均衡化处理
    if (EQUALIZE)
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));//创建一个CLAHE对象，直方图均衡算法，能有效的增强或改善图像（局部）对比度
        TicToc t_c;
        clahe->apply(_img, img);
        ROS_DEBUG("CLAHE costs: %fms", t_c.toc());
    }
    else
        img = _img;



     //2、Prev、cur、forw _img赋值并清除forw_ptr，我认为prev多余
     //cur_img ： 上一帧信息       forw_img ：当前信息
    if (forw_img.empty())
    {
        //如果当前帧的图像数据forw_img为空，说明当前是第一次读入图像数据，将读入的图像赋给当前帧forw_img
        //同时，还将读入的图像赋给prev_img、cur_img，这是为了避免后面使用到这些数据时，它们是空的
        prev_img = cur_img = forw_img = img;
    }
    else
    {
        //否则，说明之前就已经有图像读入
        //所以只需要更新当前帧forw_img的数据
        forw_img = img;
    }
    //用光流法求出来的特征点
    forw_pts.clear();//此时forw_pts还保存的是上一帧图像中的特征点，所以把它清除




    //3、光流法：根据cur_img上一帧图像、forw_img当前帧图相关、cur_ptr上一帧的2D点的矢量   光流跟踪forw_ptr  当前帧的2D点的矢量
    if (cur_pts.size() > 0)  
    //第二帧及以后的图像进行跟踪
    //cur_pts作为前一时刻已经提取出的点，送入status与err作为追踪结果的状态记录，用之加以判断追踪的结果。最后两个参数是一般默认值
    {
        TicToc t_o;
        vector<uchar> status;
        vector<float> err;
        //调用cv::calcOpticalFlowPyrLK()对前一帧的特征点cur_pts进行LK金字塔光流跟踪，得到forw_pts
        //
        //status标记了从前一帧cur_img到forw_img特征点的跟踪状态，无法被追踪到的点标记为0
        //n_pts表示存储的角点坐标集合，cur_pts表示上一帧中存储的特征点，forw_img表示在当前图像的特征点
        // status表示 cur_pts和forw_pts中对应点对是否跟踪成功
        //cv::Size(21, 21),搜索窗口的大小
        //3 金字塔层数
        cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts, status, err, cv::Size(21, 21), 3);

        //还要判断跟踪成功的角点是否都在图像内，将位于图像边界外的status点标记为0
        for (int i = 0; i < int(forw_pts.size()); i++)
            if (status[i] && !inBorder(forw_pts[i]))
                status[i] = 0;
        
        
        //跟踪失败的点，直接删除
        //4、根据status,把跟踪失败的点的所有信息去除
        // 根据跟踪状态向量去除没有跟踪到的特征点；
        //将光流跟踪后的点的集合，根据跟踪的状态(status)进行重组，并且去除没有跟踪到的点；
        reduceVector(prev_pts, status);//没用到
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        //将光流跟踪后的点的id和跟踪次数，根据跟踪的状态(status)进行重组，并且去除没有跟踪到的点；
        reduceVector(ids, status);//记录特征点id的ids，和
        reduceVector(cur_un_pts, status);//去畸变后得到坐标
        reduceVector(track_cnt, status); //记录特征点被跟踪次数的track_cnt
        ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
    }
    //光流追踪成功,特征点被成功跟踪的次数就加1
    //数值代表被追踪的次数，数值越大，说明被追踪的就越久
    //将track_cnt中的每个数进行加一处理，代表又跟踪了一次
    for (auto &n : track_cnt)
        n++;
    
     //PUB_THIS_FRAME=1 需要发布特征点
    if (PUB_THIS_FRAME)//如果发布该帧则以下操作
    {
        //5、通过基本矩阵剔除outliers
        rejectWithF(); // 根据2D特征匹配关系计算基本矩阵（对极几何），并利用rasanc算法去除外点；去除误匹配点
        ROS_DEBUG("set mask begins");
        TicToc t_m;
        // 将已有特征点的区域灰度置零，该区域就不用再检测特征点了；提供给特征检测函数用；
        //6、对特征点按追踪顺序排序，去除密集点
        setMask();//保证相邻的特征点之间要相隔30个像素,设置mask
        ROS_DEBUG("set mask costs %fms", t_m.toc());

        ROS_DEBUG("detect feature begins");
        TicToc t_t;
        
        //7、寻转新特征点补全
        int n_max_cnt = MAX_CNT - static_cast<int>(forw_pts.size());
        // 如果跟踪到的点少于设定值，补充特征，则在最新帧提取新的特征点，保证跟踪质量；
        
        if (n_max_cnt > 0)
        {
            if(mask.empty())
                cout << "mask is empty " << endl;
            if (mask.type() != CV_8UC1)
                cout << "mask type wrong " << endl;
            if (mask.size() != forw_img.size())
                cout << "wrong size " << endl;
            // 根据指定数量，指定的掩模区域 也就是黑色圆形相机周边的区域   提取新特征点，已有特征点的区域就不用再提取了，保证特征点均匀分布；在已跟踪到角点的位置上，将mask对应位置上设为0
            // forw_img表示当前图像(这里表示第一帧图像)，n_pts表示存储的角点坐标集合。MAX_CNT - forw_pts.size() 表示要检测的角点个数
            /** 
            * cv::goodFeaturesToTrack()
            * @brief   在mask中不为0的区域检测新的特征点
            * @optional    ref:https://docs.opencv.org/3.1.0/dd/d1a/group__imgproc__feature.html#ga1d6bb77486c8f92d79c8793ad995d541
            * @param[in]    InputArray _image=forw_img 输入图像
            * @param[out]   _corners=n_pts 存放检测到的角点的vector
            * @param[in]    maxCorners=MAX_CNT - forw_pts.size() 返回的角点的数量的最大值
            * @param[in]    qualityLevel=0.01 角点质量水平的最低阈值（范围为0到1，质量最高角点的水平为1），小于该阈值的角点被拒绝
            * @param[in]    minDistance=MIN_DIST 返回角点之间欧式距离的最小值
            * @param[in]    _mask=mask 和输入图像具有相同大小，类型必须为CV_8UC1,用来描述图像中感兴趣的区域，只在感兴趣区域中检测角点
            * @param[in]    blockSize：计算协方差矩阵时的窗口大小
            * @param[in]    useHarrisDetector：指示是否使用Harris角点检测，如不指定，则计算shi-tomasi角点
            * @param[in]    harrisK：Harris角点检测需要的k值
            * @return      void
            */
            cv::goodFeaturesToTrack(forw_img, n_pts, MAX_CNT - forw_pts.size(), 0.01, MIN_DIST, mask);
        }
        else
            n_pts.clear();
        ROS_DEBUG("detect feature costs: %fms", t_t.toc());

        ROS_DEBUG("add feature begins");
        TicToc t_a;
        
        //8、添将新检测到的特征点n_pts添加到forw_pts中，id初始化-1,track_cnt初始化为1.
        // 将新提取的特征点加入到跟踪向量中； cv::goodFeaturesToTrack里检测到角点(n_pts)后，将n_pts中的角点放到forw_pts中，ids表示每个角点的编号，track_cnt表示每个角点的跟踪次数
        addPoints();
        ROS_DEBUG("selectFeature costs: %fms", t_a.toc());
    }
    //9、滑到下一图像
    //当下一帧图像到来时，当前帧数据就成为了上一帧发布的数据
    prev_img = cur_img;//在第一帧处理中还是等于当前帧forw_img
    prev_pts = cur_pts;//在第一帧中不做处理
    prev_un_pts = cur_un_pts;//在第一帧中不做处理
    cur_img = forw_img;//将当前帧赋值给上一帧
    cur_pts = forw_pts;
    // 根据相机模型去除视觉特征畸变，对于跟踪的特征进行去畸变，并计算特征运动速度；
    
    //10、根据不同的相机模型进行去畸变矫正和深度归一化，计算速度
    undistortedPoints();
    prev_time = cur_time;
}

//去除误匹配的点
void FeatureTracker::rejectWithF()
{
    if (forw_pts.size() >= 8)//取大于8帧以上的情况
    {
        ROS_DEBUG("FM ransac begins");
        TicToc t_f;
        vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_forw_pts(forw_pts.size());
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            Eigen::Vector3d tmp_p;//应用了一个虚拟相机
            m_camera->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera->liftProjective(Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_forw_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        vector<uchar> status;
        cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        int size_a = cur_pts.size();
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, forw_pts.size(), 1.0 * forw_pts.size() / size_a);
        ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
}

//更新序号，新加入的特征点更新全局id
bool FeatureTracker::updateID(unsigned int i)
{
    if (i < ids.size())//n_id是static类型的数据，具有累加的功能
    {
        if (ids[i] == -1)
            ids[i] = n_id++;
        return true;
    }
    else
        return false;
}

void FeatureTracker::readIntrinsicParameter(const string &calib_file)
{
    ROS_INFO("reading paramerter of camera %s", calib_file.c_str());
    m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
}

void FeatureTracker::showUndistortion(const string &name)
{
    cv::Mat undistortedImg(ROW + 600, COL + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < COL; i++)
        for (int j = 0; j < ROW; j++)
        {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
            //printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
        }
    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + COL / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + ROW / 2;
        pp.at<float>(2, 0) = 1.0;
        //cout << trackerData[0].K << endl;
        //printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        //printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < ROW + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < COL + 600)
        {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else
        {
            //ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
    cv::imshow(name, undistortedImg);
    cv::waitKey(0);
}


// 根据相机模型去除视觉特征畸变，对于跟踪的特征进行去畸变，并计算特征运动速度；
void FeatureTracker::undistortedPoints()
{
    cur_un_pts.clear();
    cur_un_pts_map.clear();
    //cv::undistortPoints(cur_pts, un_pts, K, cv::Mat());
    for (unsigned int i = 0; i < cur_pts.size(); i++)
    {
        Eigen::Vector2d a(cur_pts[i].x, cur_pts[i].y);
        Eigen::Vector3d b;
        // 根据不同的相机模型将二维坐标转换到三维坐标
        m_camera->liftProjective(a, b);
        // 再延伸到深度归一化平面上
        cur_un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
        cur_un_pts_map.insert(make_pair(ids[i], cv::Point2f(b.x() / b.z(), b.y() / b.z())));
        //printf("cur pts id %d %f %f", ids[i], cur_un_pts[i].x, cur_un_pts[i].y);
    }
    // caculate points velocity,
    //计算速度，给imu标定和时间偏移做准备
    // 计算每个特征点的速度到pts_velocity
    if (!prev_un_pts_map.empty())
    {
        double dt = cur_time - prev_time;
        pts_velocity.clear();
        for (unsigned int i = 0; i < cur_un_pts.size(); i++)
        {
            if (ids[i] != -1)
            {
                std::map<int, cv::Point2f>::iterator it;
                it = prev_un_pts_map.find(ids[i]);
                if (it != prev_un_pts_map.end())
                {
                    double v_x = (cur_un_pts[i].x - it->second.x) / dt;
                    double v_y = (cur_un_pts[i].y - it->second.y) / dt;
                    pts_velocity.push_back(cv::Point2f(v_x, v_y));
                }
                else
                    pts_velocity.push_back(cv::Point2f(0, 0));
            }
            else
            {
                pts_velocity.push_back(cv::Point2f(0, 0));
            }
        }
    }
    else
    {
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    prev_un_pts_map = cur_un_pts_map;
}
