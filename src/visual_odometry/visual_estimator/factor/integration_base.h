#pragma once

#include "../utility/utility.h"
#include "../parameters.h"

#include <ceres/ceres.h>
using namespace Eigen;

/**
* @class IntegrationBase IMU pre-integration class
* @Description 
*/
 //预积分类
class IntegrationBase
{
  public:

    double dt;
    Eigen::Vector3d acc_0, gyr_0;
    Eigen::Vector3d acc_1, gyr_1;

    const Eigen::Vector3d linearized_acc, linearized_gyr;
    Eigen::Vector3d linearized_ba, linearized_bg;

    Eigen::Matrix<double, 15, 15> jacobian;
    Eigen::Matrix<double, 15, 15> covariance;
    Eigen::Matrix<double, 18, 18> noise;

    double sum_dt;
    Eigen::Vector3d delta_p;
    Eigen::Quaterniond delta_q;
    Eigen::Vector3d delta_v;

    // saves all the IMU measurements and time difference between two image frames
    std::vector<double> dt_buf;
    std::vector<Eigen::Vector3d> acc_buf;
    std::vector<Eigen::Vector3d> gyr_buf;

    IntegrationBase() = delete;

    //预积分类  抽象传感器
    IntegrationBase(const Eigen::Vector3d &_acc_0, 
                    const Eigen::Vector3d &_gyr_0,
                    const Eigen::Vector3d &_linearized_ba, 
                    const Eigen::Vector3d &_linearized_bg)
        : acc_0{_acc_0}, 
          gyr_0{_gyr_0}, 
          linearized_acc{_acc_0}, 
          linearized_gyr{_gyr_0},
          linearized_ba{_linearized_ba}, 
          linearized_bg{_linearized_bg},
          jacobian{Eigen::Matrix<double, 15, 15>::Identity()}, //雅阁比初始化为单位阵
          covariance{Eigen::Matrix<double, 15, 15>::Zero()},//协方差初始化为零阵
          sum_dt{0.0}, 
          delta_p{Eigen::Vector3d::Zero()}, //P初始化为零向量
          delta_q{Eigen::Quaterniond::Identity()}, //Q初始化为单位四元数
          delta_v{Eigen::Vector3d::Zero()}//V初始化为零向量

    {
        //初始化为Q噪声项的对角协方差矩阵、
        noise = Eigen::Matrix<double, 18, 18>::Zero();
        noise.block<3, 3>(0, 0) =  (ACC_N * ACC_N) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(3, 3) =  (GYR_N * GYR_N) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(6, 6) =  (ACC_N * ACC_N) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(9, 9) =  (GYR_N * GYR_N) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(12, 12) =  (ACC_W * ACC_W) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(15, 15) =  (GYR_W * GYR_W) * Eigen::Matrix3d::Identity();
    }

    //
    void push_back(double dt, const Eigen::Vector3d &acc, const Eigen::Vector3d &gyr)
    {
        dt_buf.push_back(dt);
        acc_buf.push_back(acc);
        gyr_buf.push_back(gyr);
        propagate(dt, acc, gyr);
    }

    // after optimization, repropagate pre-integration using the updated bias
    void repropagate(const Eigen::Vector3d &_linearized_ba, const Eigen::Vector3d &_linearized_bg)
    {
        sum_dt = 0.0;
        acc_0 = linearized_acc;
        gyr_0 = linearized_gyr;
        delta_p.setZero();
        delta_q.setIdentity();
        delta_v.setZero();
        linearized_ba = _linearized_ba;
        linearized_bg = _linearized_bg;
        jacobian.setIdentity();
        covariance.setZero();
        for (int i = 0; i < static_cast<int>(dt_buf.size()); i++)
            propagate(dt_buf[i], acc_buf[i], gyr_buf[i]);
    }

    /**
    * @brief   IMU预积分传播方程
    * @Description  积分计算两个关键帧之间IMU测量的变化量： 
    *               旋转delta_q 速度delta_v 位移delta_p
    *               加速度的biaslinearized_ba 陀螺仪的Bias linearized_bg
    *               同时维护更新预积分的Jacobian和Covariance,计算优化时必要的参数
    * @param[in]   _dt 时间间隔
    * @param[in]   _acc_1 线加速度
    * @param[in]   _gyr_1 角速度
    * @return  void
    */
    void propagate(double _dt, const Eigen::Vector3d &_acc_1, const Eigen::Vector3d &_gyr_1)
    {

        dt = _dt;
        acc_1 = _acc_1;
        gyr_1 = _gyr_1;
        //定义结果变量
        Vector3d result_delta_p;
        Quaterniond result_delta_q;
        Vector3d result_delta_v;
        Vector3d result_linearized_ba;
        Vector3d result_linearized_bg;

        midPointIntegration(_dt, acc_0, gyr_0, _acc_1, _gyr_1, delta_p, delta_q, delta_v,
                            linearized_ba, linearized_bg,
                            result_delta_p, result_delta_q, result_delta_v,
                            result_linearized_ba, result_linearized_bg, 1);

        delta_p = result_delta_p;
        delta_q = result_delta_q;
        delta_v = result_delta_v;
        linearized_ba = result_linearized_ba;
        linearized_bg = result_linearized_bg;
        delta_q.normalize();
        sum_dt += dt;
        acc_0 = acc_1;
        gyr_0 = gyr_1;  
     
    }

    /**
    * @brief   IMU预积分中采用中值积分递推Jacobian和Covariance
    *          构造误差的线性化递推方程，得到Jacobian和Covariance递推公式-> Paper 式9、10、11
    * @return  void
    */
    void midPointIntegration(double _dt, 
                            const Eigen::Vector3d &_acc_0, const Eigen::Vector3d &_gyr_0,
                            const Eigen::Vector3d &_acc_1, const Eigen::Vector3d &_gyr_1,
                            const Eigen::Vector3d &delta_p, const Eigen::Quaterniond &delta_q, const Eigen::Vector3d &delta_v,
                            const Eigen::Vector3d &linearized_ba, const Eigen::Vector3d &linearized_bg,
                            Eigen::Vector3d &result_delta_p, Eigen::Quaterniond &result_delta_q, Eigen::Vector3d &result_delta_v,
                            Eigen::Vector3d &result_linearized_ba, Eigen::Vector3d &result_linearized_bg, bool update_jacobian)
    {
        //ROS_INFO("midpoint integration");
        //预积分量，其实没有实际物理意义，是预积分估计量
        Vector3d un_acc_0 = delta_q * (_acc_0 - linearized_ba);
        Vector3d un_gyr = 0.5 * (_gyr_0 + _gyr_1) - linearized_bg;
        result_delta_q = delta_q * Quaterniond(1, un_gyr(0) * _dt / 2, un_gyr(1) * _dt / 2, un_gyr(2) * _dt / 2);
        Vector3d un_acc_1 = result_delta_q * (_acc_1 - linearized_ba);
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        result_delta_p = delta_p + delta_v * _dt + 0.5 * un_acc * _dt * _dt;
        result_delta_v = delta_v + un_acc * _dt;
        result_linearized_ba = linearized_ba;
        result_linearized_bg = linearized_bg;         

        // 为什么需要此处的雅克比矩阵，即预积分值对各个状态误差的导数？
        // 优化的时候需要用到当前预积分值的协方差，而要推导预积分量的协方差,我们需要知道 imu 噪声和预积分量之间的线性递推关系；
        // 其中IMU单次测量误差可以由标定的参数得到，而预积分是一段时间的IMU数据累积，其误差需要通过误差递推得到；
        // 误差的传递由两部分组成:当前时刻的误差传递给下一时刻,当前时刻测量噪声传递给下一时刻；
        // 由于IMU积分过程中涉及到四元数及位姿的计算，是一个非线性过程，我们可以基于一阶泰勒展开将递推过程线性化；
        // 预积分值由真值和误差组成，其中-->
        // --> 预积分误差是由 各个状态量误差 以及 噪声 引起的，因此我们只要推导出 预积分量 对 各个状态量误差 的导数，
        // 即每个状态量产生一点儿误差，会对预积分值产生多大误差；然后我们就可以根据当前的状态量误差算出当前预积分的误差，进而计算协方差；
        // (此句结论需要反复重点理解，这是求误差递推的大方向，以及为什么要求各个状态量误差的雅克比)
        
        
        //更新pvq增量误差下的雅阁比矩阵
        if(update_jacobian)
        {
            Vector3d w_x = 0.5 * (_gyr_0 + _gyr_1) - linearized_bg;
            Vector3d a_0_x = _acc_0 - linearized_ba;
            Vector3d a_1_x = _acc_1 - linearized_ba;
            Matrix3d R_w_x, R_a_0_x, R_a_1_x;

            //反对称矩阵
            R_w_x<<0, -w_x(2), w_x(1),
                w_x(2), 0, -w_x(0),
                -w_x(1), w_x(0), 0;
            R_a_0_x<<0, -a_0_x(2), a_0_x(1),
                a_0_x(2), 0, -a_0_x(0),
                -a_0_x(1), a_0_x(0), 0;
            R_a_1_x<<0, -a_1_x(2), a_1_x(1),
                a_1_x(2), 0, -a_1_x(0),
                -a_1_x(1), a_1_x(0), 0;

            //F矩阵
            MatrixXd F = MatrixXd::Zero(15, 15);
            F.block<3, 3>(0, 0) = Matrix3d::Identity();
            //f01
            F.block<3, 3>(0, 3) = -0.25 * delta_q.toRotationMatrix() * R_a_0_x * _dt * _dt + 
                                  -0.25 * result_delta_q.toRotationMatrix() * R_a_1_x * (Matrix3d::Identity() - R_w_x * _dt) * _dt * _dt;
            //δt
            F.block<3, 3>(0, 6) = MatrixXd::Identity(3,3) * _dt;
            //f03
            F.block<3, 3>(0, 9) = -0.25 * (delta_q.toRotationMatrix() + result_delta_q.toRotationMatrix()) * _dt * _dt;
            //f04
            F.block<3, 3>(0, 12) = -0.25 * result_delta_q.toRotationMatrix() * R_a_1_x * _dt * _dt * -_dt;
           
            F.block<3, 3>(3, 3) = Matrix3d::Identity() - R_w_x * _dt;
            F.block<3, 3>(3, 12) = -1.0 * MatrixXd::Identity(3,3) * _dt;
            F.block<3, 3>(6, 3) = -0.5 * delta_q.toRotationMatrix() * R_a_0_x * _dt + 
                                  -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x * (Matrix3d::Identity() - R_w_x * _dt) * _dt;
            F.block<3, 3>(6, 6) = Matrix3d::Identity();
            F.block<3, 3>(6, 9) = -0.5 * (delta_q.toRotationMatrix() + result_delta_q.toRotationMatrix()) * _dt;
            F.block<3, 3>(6, 12) = -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x * _dt * -_dt;
            F.block<3, 3>(9, 9) = Matrix3d::Identity();
            F.block<3, 3>(12, 12) = Matrix3d::Identity();
            //cout<<"A"<<endl<<A<<endl;

            MatrixXd V = MatrixXd::Zero(15,18);
            V.block<3, 3>(0, 0) =  0.25 * delta_q.toRotationMatrix() * _dt * _dt;
            V.block<3, 3>(0, 3) =  0.25 * -result_delta_q.toRotationMatrix() * R_a_1_x  * _dt * _dt * 0.5 * _dt;
            V.block<3, 3>(0, 6) =  0.25 * result_delta_q.toRotationMatrix() * _dt * _dt;
            V.block<3, 3>(0, 9) =  V.block<3, 3>(0, 3);
            V.block<3, 3>(3, 3) =  0.5 * MatrixXd::Identity(3,3) * _dt;
            V.block<3, 3>(3, 9) =  0.5 * MatrixXd::Identity(3,3) * _dt;
            V.block<3, 3>(6, 0) =  0.5 * delta_q.toRotationMatrix() * _dt;
            V.block<3, 3>(6, 3) =  0.5 * -result_delta_q.toRotationMatrix() * R_a_1_x  * _dt * 0.5 * _dt;
            V.block<3, 3>(6, 6) =  0.5 * result_delta_q.toRotationMatrix() * _dt;
            V.block<3, 3>(6, 9) =  V.block<3, 3>(6, 3);
            V.block<3, 3>(9, 12) = MatrixXd::Identity(3,3) * _dt;
            V.block<3, 3>(12, 15) = MatrixXd::Identity(3,3) * _dt;

            // 其中,Jacobian 的初始值为Jk = I，这里计算出来的Jk+1 只是为了给后面提供对 bias 的Jacobian。
            jacobian = F * jacobian;

            // 其中,协方差初始值P k = 0； Q为表示噪声项的对角协方差矩阵，
            // 由于噪声之间相互独立，Q矩阵非对角线元素均为零，对角线元素为标定得到的加速度计和陀螺仪的各个轴的噪声的平方；
            covariance = F * covariance * F.transpose() + V * noise * V.transpose();
        }

    }

    
    // calculate residuals for ceres optimization, used in imu_factor.h
    // paper equation 24
    // 计算待优化变量与IMU测量值之间的误差；
    Eigen::Matrix<double, 15, 1> evaluate(const Eigen::Vector3d &Pi, const Eigen::Quaterniond &Qi, const Eigen::Vector3d &Vi, const Eigen::Vector3d &Bai, const Eigen::Vector3d &Bgi,
                                          const Eigen::Vector3d &Pj, const Eigen::Quaterniond &Qj, const Eigen::Vector3d &Vj, const Eigen::Vector3d &Baj, const Eigen::Vector3d &Bgj)
    {
        Eigen::Matrix<double, 15, 1> residuals;

        // 取出IMU的各个预积分值对bias误差的导数(雅克比矩阵)
        Eigen::Matrix3d dp_dba = jacobian.block<3, 3>(O_P, O_BA);
        Eigen::Matrix3d dp_dbg = jacobian.block<3, 3>(O_P, O_BG);

        Eigen::Matrix3d dq_dbg = jacobian.block<3, 3>(O_R, O_BG);

        Eigen::Matrix3d dv_dba = jacobian.block<3, 3>(O_V, O_BA);
        Eigen::Matrix3d dv_dbg = jacobian.block<3, 3>(O_V, O_BG);

        // 计算当前预积分段相比于上次的bias的变化量，即bias的误差;
        Eigen::Vector3d dba = Bai - linearized_ba;
        Eigen::Vector3d dbg = Bgi - linearized_bg;

        // 已知IMU的各个预积分值对bias的导数(雅克比矩阵)，已知两帧之间bias的变化量，
        // 计算由bias的变化导致的pvq的变化量，然后将其叠加到原来的预积分值上进行补偿，得到更精确的预积分值；
        Eigen::Quaterniond corrected_delta_q = delta_q * Utility::deltaQ(dq_dbg * dbg);
        Eigen::Vector3d corrected_delta_v = delta_v + dv_dba * dba + dv_dbg * dbg;
        Eigen::Vector3d corrected_delta_p = delta_p + dp_dba * dba + dp_dbg * dbg;

        // 根据优化变量相邻帧之间的位姿增量和IMU预积分位姿增量计算残差；(由于重力向量的约束，使得横滚俯仰角是可观的)
        residuals.block<3, 1>(O_P, 0) = Qi.inverse() * (0.5 * G * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt) - corrected_delta_p;
        residuals.block<3, 1>(O_R, 0) = 2 * (corrected_delta_q.inverse() * (Qi.inverse() * Qj)).vec();
        residuals.block<3, 1>(O_V, 0) = Qi.inverse() * (G * sum_dt + Vj - Vi) - corrected_delta_v;
        residuals.block<3, 1>(O_BA, 0) = Baj - Bai;
        residuals.block<3, 1>(O_BG, 0) = Bgj - Bgi;
        return residuals;
    }
};