#include "feature_manager.h"

int FeaturePerId::endFrame()
{
    return start_frame + feature_per_frame.size() - 1;
}

FeatureManager::FeatureManager(Matrix3d _Rs[])
    : Rs(_Rs)
{
    for (int i = 0; i < NUM_OF_CAM; i++)
        ric[i].setIdentity();
}

void FeatureManager::setRic(Matrix3d _ric[])
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ric[i] = _ric[i];
    }
}

void FeatureManager::clearState()
{
    feature.clear();
}

int FeatureManager::getFeatureCount()
{
    int cnt = 0;
    for (auto &it : feature)
    {

        it.used_num = it.feature_per_frame.size();

        if (it.used_num >= 2 && it.start_frame < WINDOW_SIZE - 2)
        {
            cnt++;
        }
    }
    return cnt;
}

// 迭代当前帧检测到的每个路标点 id,看看feature队列中是否包含,若不含,则添加到feature队列中;若包含,则添加到对应id的FeaturePerFrame队列。
// 计算第2最新帧与第3最新帧之间的平均视差（当前帧是第1最新帧），然后判断是否把第2最新帧添加为关键帧
bool FeatureManager::addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 8, 1>>>> &image, double td)
{
    ROS_DEBUG("input feature: %d", (int)image.size());
    ROS_DEBUG("num of feature: %d", getFeatureCount());
    double parallax_sum = 0; // 视差总和
    int parallax_num = 0;    // 计算视差的特征点数, 用于计算平均视差
    last_track_num = 0;      // 跟踪上的特征点数
    for (auto &id_pts : image)
    {
        FeaturePerFrame f_per_fra(id_pts.second[0].second, td);

        // find feature id in the feature bucket
        int feature_id = id_pts.first;
        //在滑动窗口进行遍历查找
        auto it = find_if(feature.begin(), feature.end(), [feature_id](const FeaturePerId &it)
                          {return it.feature_id == feature_id;});
        //没有找到就是新的特征点，增加一些信息
        if (it == feature.end())
        {
            // 第一次观察到图像中的这个特征，创建一个新的特征对象
            feature.push_back(FeaturePerId(feature_id, frame_count, f_per_fra.depth));
            feature.back().feature_per_frame.push_back(f_per_fra);//特征也push进去
        }
        else if (it->feature_id == feature_id)//找到了
        {
            //之前已经观察到图像中的这个特征
            it->feature_per_frame.push_back(f_per_fra);
            last_track_num++;
            // 有时特征在没有深度的情况下首先被观察到
             //（如果相机移动很快，用当前图像深度初始化初始特征深度并不完全准确，那么可以注释掉下面的线条）
            //新增
            if (f_per_fra.depth > 0 && it->lidar_depth_flag == false)
            {
                it->estimated_depth = f_per_fra.depth;
                it->lidar_depth_flag = true;
                it->feature_per_frame[0].depth = f_per_fra.depth;
            }
        }
    }
    //滑动窗口
    if (frame_count < 2 || last_track_num < 20)
        return true; // marge最老帧

    for (auto &it_per_id : feature)
    {   //次次新帧
        if (it_per_id.start_frame <= frame_count - 2 &&
            it_per_id.start_frame + int(it_per_id.feature_per_frame.size()) - 1 >= frame_count - 1)
        {
            // 计算特征点的视差 (倒数第二帧 和 倒数第三帧 之间的视差)
            parallax_sum += compensatedParallax2(it_per_id, frame_count);
            parallax_num++;
        }
    }
    // 没有同时观测到一个特征点
    if (parallax_num == 0)
    {
        return true;
    }
    else
    {
        // 根据平均视差决定marge最老帧还是次新帧
        ROS_DEBUG("parallax_sum: %lf, parallax_num: %d", parallax_sum, parallax_num);
        ROS_DEBUG("current parallax: %lf", parallax_sum / parallax_num * FOCAL_LENGTH);
        return parallax_sum / parallax_num >= MIN_PARALLAX;
    }
}

void FeatureManager::debugShow()
{
    ROS_DEBUG("debug show");
    for (auto &it : feature)
    {
        ROS_ASSERT(it.feature_per_frame.size() != 0);
        ROS_ASSERT(it.start_frame >= 0);
        ROS_ASSERT(it.used_num >= 0);

        ROS_DEBUG("%d,%d,%d ", it.feature_id, it.used_num, it.start_frame);
        int sum = 0;
        for (auto &j : it.feature_per_frame)
        {
            ROS_DEBUG("%d,", int(j.is_used));
            sum += j.is_used;
            printf("(%lf,%lf) ",j.point(0), j.point(1));
        }
        ROS_ASSERT(it.used_num == sum);
    }
}

//旋转外参标定
vector<pair<Vector3d, Vector3d>> FeatureManager::getCorresponding(int frame_count_l, int frame_count_r)
{
    vector<pair<Vector3d, Vector3d>> corres;
    for (auto &it : feature)
    {
        if (it.start_frame <= frame_count_l && it.endFrame() >= frame_count_r)
        {
            Vector3d a = Vector3d::Zero(), b = Vector3d::Zero();
            int idx_l = frame_count_l - it.start_frame;
            int idx_r = frame_count_r - it.start_frame;

            a = it.feature_per_frame[idx_l].point;

            b = it.feature_per_frame[idx_r].point;
            
            corres.push_back(make_pair(a, b));
        }
    }
    return corres;
}

void FeatureManager::setDepth(const VectorXd &x)
{
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        it_per_id.estimated_depth = 1.0 / x(++feature_index);

        if (it_per_id.estimated_depth < 0)
        {
            it_per_id.solve_flag = 2;
        }
        else
            it_per_id.solve_flag = 1;
    }
}

void FeatureManager::removeFailures()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        if (it->solve_flag == 2)
            feature.erase(it);
    }
}

void FeatureManager::clearDepth(const VectorXd &x)
{
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        it_per_id.estimated_depth = 1.0 / x(++feature_index);
        it_per_id.lidar_depth_flag = false;
    }
}

VectorXd FeatureManager::getDepthVector()
{
    VectorXd dep_vec(getFeatureCount());
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        // optimized depth after ceres maybe negative, initialize them with default value for this optimization
        if (it_per_id.estimated_depth > 0)
            dep_vec(++feature_index) = 1. / it_per_id.estimated_depth;
        else
            dep_vec(++feature_index) = 1. / INIT_DEPTH;
    }
    return dep_vec;
}

// 利用初始位姿，对每帧图像还未具有深度信息的特征点进行三角化；构建重投影误差
void FeatureManager::triangulate(Vector3d Ps[], Vector3d tic[], Matrix3d ric[])
{
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();//每被观测到一次就往里面赠一个数据，观测数量
        // 只有特征的观测次数和观测时间均超过2，才能较好的通过视差三角化出深度；
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))//条件
            continue;

        // depth is available, skip triangulation (trust the first estimate)
        if (it_per_id.estimated_depth > 0)
            continue;

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

        ROS_ASSERT(NUM_OF_CAM == 1);
        Eigen::MatrixXd svd_A(2 * it_per_id.feature_per_frame.size(), 4);
        int svd_idx = 0;

        Eigen::Matrix<double, 3, 4> P0;
        // 将当前 pose 转换到左目相机相对于世界坐标系下；（默认是imu到world）
        Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];
        Eigen::Matrix3d R0 = Rs[imu_i] * ric[0]; 
        P0.leftCols<3>() = Eigen::Matrix3d::Identity();
        P0.rightCols<1>() = Eigen::Vector3d::Zero();

        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            //三角化
            Eigen::Vector3d t1 = Ps[imu_j] + Rs[imu_j] * tic[0];
            Eigen::Matrix3d R1 = Rs[imu_j] * ric[0];
            Eigen::Vector3d t = R0.transpose() * (t1 - t0);
            Eigen::Matrix3d R = R0.transpose() * R1;
            Eigen::Matrix<double, 3, 4> P;
            // 相机相邻位姿矩阵
            P.leftCols<3>() = R.transpose();
            P.rightCols<1>() = -R.transpose() * t;
            Eigen::Vector3d f = it_per_frame.point.normalized();
            svd_A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0);
            svd_A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);//svd分解

            if (imu_i == imu_j)
                continue;
        }
        ROS_ASSERT(svd_idx == svd_A.rows());
        Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
        double svd_method = svd_V[2] / svd_V[3];

        // update depth from triangulation
        it_per_id.estimated_depth = svd_method;
        // check if triangulation failed
        if (it_per_id.estimated_depth < 0)
        {
            it_per_id.estimated_depth = INIT_DEPTH;
        }
    }
}

void FeatureManager::removeOutlier()
{
    ROS_BREAK();
    int i = -1;
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        i += it->used_num != 0;
        if (it->used_num != 0 && it->is_outlier == true)
        {
            feature.erase(it);
        }
    }
}

void FeatureManager::removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P)
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            // feature point and depth in old local camera frame
            Eigen::Vector3d uv_i = it->feature_per_frame[0].point;
            double depth = -1;
            if (it->feature_per_frame[0].depth > 0)
                // if lidar depth available at this frame for feature
                depth = it->feature_per_frame[0].depth;
            else if (it->estimated_depth > 0)
                // if estimated depth available
                depth = it->estimated_depth;

            // delete current feature in the old local camera frame
            it->feature_per_frame.erase(it->feature_per_frame.begin());

            if (it->feature_per_frame.size() < 2)
            {
                // delete feature from feature manager
                feature.erase(it);
                continue;
            }
            else
            {
                Eigen::Vector3d pts_i = uv_i * depth; // feature in cartisian space in old local camera frame
                Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P; // feautre in cartisian space in world frame
                Eigen::Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P); // feature in cartisian space in shifted local camera frame
                double dep_j = pts_j(2);

                // after deletion, the feature has lidar depth in the first of the remaining frame
                if (it->feature_per_frame[0].depth > 0)
                {
                    it->estimated_depth = it->feature_per_frame[0].depth;
                    it->lidar_depth_flag = true;
                } 
                // calculated depth in the current frame
                else if (dep_j > 0)
                {
                    it->estimated_depth = dep_j;
                    it->lidar_depth_flag = false;
                } 
                // non-positive depth, invalid
                else 
                {
                    it->estimated_depth = INIT_DEPTH;
                    it->lidar_depth_flag = false;
                }
            }
        }
    }
}

void FeatureManager::removeBack()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            if (it->feature_per_frame.size() == 0)
                feature.erase(it);
        }
    }
}

void FeatureManager::removeFront(int frame_count)
{
    for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame == frame_count)
        {
            it->start_frame--;
        }
        else
        {
            int j = WINDOW_SIZE - 1 - it->start_frame;
            if (it->endFrame() < frame_count - 1)
                continue;
            it->feature_per_frame.erase(it->feature_per_frame.begin() + j);
            if (it->feature_per_frame.size() == 0)
                feature.erase(it);
        }
    }
}

// 计算特征点的视差 (倒数第二帧 和 倒数第三帧 之间的视差)
double FeatureManager::compensatedParallax2(const FeaturePerId &it_per_id, int frame_count)
{
    //check the second last frame is keyframe or not
    //parallax betwwen seconde last frame and third last frame
    const FeaturePerFrame &frame_i = it_per_id.feature_per_frame[frame_count - 2 - it_per_id.start_frame];
    const FeaturePerFrame &frame_j = it_per_id.feature_per_frame[frame_count - 1 - it_per_id.start_frame];

    double ans = 0;
    Vector3d p_j = frame_j.point;

    double u_j = p_j(0);
    double v_j = p_j(1);

    Vector3d p_i = frame_i.point;
    Vector3d p_i_comp;

    //int r_i = frame_count - 2;
    //int r_j = frame_count - 1;
    //p_i_comp = ric[camera_id_j].transpose() * Rs[r_j].transpose() * Rs[r_i] * ric[camera_id_i] * p_i;
    p_i_comp = p_i;
    double dep_i = p_i(2);
    double u_i = p_i(0) / dep_i;
    double v_i = p_i(1) / dep_i;
    double du = u_i - u_j, dv = v_i - v_j;

    double dep_i_comp = p_i_comp(2);
    double u_i_comp = p_i_comp(0) / dep_i_comp;
    double v_i_comp = p_i_comp(1) / dep_i_comp;
    double du_comp = u_i_comp - u_j, dv_comp = v_i_comp - v_j;

    ans = max(ans, sqrt(min(du * du + dv * dv, du_comp * du_comp + dv_comp * dv_comp)));

    return ans;
}