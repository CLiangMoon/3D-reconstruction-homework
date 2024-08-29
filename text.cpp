#include <iostream>
#include <vector>
#include <cmath>
#include <cfloat>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/SVD>

class SingleCamera {
public:
    SingleCamera(Eigen::MatrixXf world_coor, Eigen::MatrixXf pixel_coor, int n)
        : world_coor(world_coor), pixel_coor(pixel_coor), point_num(n),
          P(Eigen::MatrixXf::Zero(2*n, 12)), M(Eigen::MatrixXf::Zero(3, 4)),
          A(Eigen::MatrixXf::Zero(3, 3)), b(Eigen::MatrixXf::Zero(3, 1)),
          K(Eigen::MatrixXf::Zero(3, 3)), R(Eigen::MatrixXf::Zero(3, 3)),
          t(Eigen::MatrixXf::Zero(3, 1)) {}

    void composeP();
    void svdP();
    void workIntrinsicAndExtrinsic();
    void selfcheck(const Eigen::MatrixXf& w_check, const Eigen::MatrixXf& c_check);

private:
    Eigen::MatrixXf world_coor;
    Eigen::MatrixXf pixel_coor;
    int point_num;

    Eigen::MatrixXf P;
    Eigen::MatrixXf M;
    Eigen::MatrixXf A;
    Eigen::MatrixXf b;
    Eigen::MatrixXf K;
    Eigen::MatrixXf R;
    Eigen::MatrixXf t;
};

void SingleCamera::composeP() {
    for (int i = 0; i < point_num; ++i) {
        P.row(i) << world_coor(i, 0), world_coor(i, 1), world_coor(i, 2), 1, 
                    0, 0, 0, 0, 
                    -pixel_coor(i, 0) * world_coor(i, 0), -pixel_coor(i, 0) * world_coor(i, 1), -pixel_coor(i, 0) * world_coor(i, 2),-pixel_coor(i, 0);
        P.row(i + point_num) << 0, 0, 0, 0, 
                                world_coor(i, 0), world_coor(i, 1), world_coor(i, 2), 1, 
                                -pixel_coor(i, 1) * world_coor(i, 0), -pixel_coor(i, 1) * world_coor(i, 1), -pixel_coor(i, 1) * world_coor(i, 2),-pixel_coor(i, 1);
    }
    cout<<"P:\n"<<P<<endl;
}

void SingleCamera::svdP() {
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(P, Eigen::ComputeFullV);
    Eigen::VectorXf singular_values = svd.singularValues();
    M = svd.matrixV().col(11).head(12);
    M.resize(3, 4);
    A = M.block<3, 3>(0, 0);
    b = M.col(3);
    cout<<"M:\n"<<M<<endl;
    cout<<"A:\n"<<A<<endl;
    cout<<"b:\n"<<b.transpose()<<endl;
}

void SingleCamera::workIntrinsicAndExtrinsic() {
    Eigen::MatrixXf A_inv = A.inverse();
    K = A.block<3, 3>(0, 0);
    R = A_inv * M.block<3, 3>(0, 0);
    t = -R * b;
    
    std::cout << "K is " << std::endl << K << std::endl;
    std::cout << "R is " << std::endl << R << std::endl;
    std::cout << "t is " << std::endl << t.transpose() << std::endl;
}

void SingleCamera::selfcheck(const Eigen::MatrixXf& w_check, const Eigen::MatrixXf& c_check) {
    float average_err = 0.0;
    
    for (int i = 0; i < w_check.rows(); ++i) {
        Eigen::Vector4f world_point = w_check.row(i).transpose();
        Eigen::Vector3f pixel_point = K * (R * world_point.head<3>() + t);
        pixel_point /= pixel_point(2);
        
        float error = (pixel_point.head<2>() - c_check.row(i).transpose()).norm();
        average_err += error;
    }
    
    average_err /= w_check.rows();
    std::cout << "The average error is " << average_err << "," << std::endl;
    if (average_err > 0.1) {
        std::cout << "which is more than 0.1" << std::endl;
    } else {
        std::cout << "which is smaller than 0.1, the M is acceptable" << std::endl;
    }
}

int main(int argc, char ** argv) {
    Eigen::MatrixXf w_xz(4, 4);
    w_xz << 8, 0, 9, 1,
            8, 0, 1, 1,
            6, 0, 1, 1,
            6, 0, 9, 1;

    Eigen::MatrixXf w_xy(4, 4);
    w_xy << 5, 1, 0, 1,
            5, 9, 0, 1,
            4, 9, 0, 1,
            4, 1, 0, 1;

    Eigen::MatrixXf w_yz(4, 4);
    w_yz << 0, 4, 7, 1,
            0, 4, 3, 1,
            0, 8, 3, 1,
            0, 8, 7, 1;

    Eigen::MatrixXf w_coor(12, 4);
    w_coor << w_xz,
            w_xy,
            w_yz;

    Eigen::MatrixXf c_xz(4, 2);
    c_xz << 275, 142,
            312, 454,
            382, 436,
            357, 134;

    Eigen::MatrixXf c_xy(4, 2);
    c_xy << 432, 473,
            612, 623,
            647, 606,
            464, 465;

    Eigen::MatrixXf c_yz(4, 2);
    c_yz << 654, 216,
            644, 368,
            761, 420,
            781, 246;

    Eigen::MatrixXf c_coor(12, 2);
    c_coor << c_xz,
            c_xy,
            c_yz;

    Eigen::MatrixXf w_check(5, 4);
    w_check << 6, 0, 5, 1,
                3, 3, 0, 1,
                0, 4, 0, 1,
                0, 4, 4, 1,
                0, 0, 7, 1;

    Eigen::MatrixXf c_check(5, 2);
    c_check << 369, 297,
                531, 484,
                640, 468,
                646, 333,
                556, 194;

    SingleCamera aCamera = SingleCamera(w_coor, c_coor, 12);  // 12 points in total are used
    aCamera.composeP();
    aCamera.svdP();
    aCamera.workIntrinsicAndExtrinsic();
    aCamera.selfcheck(w_check, c_check);  // test 5 points and verify M

    return 0;
}

//在运行克制执行程序的时候出现了报错，错误原因是Too few coefficients passed to comma initializer
//正好练习一下程序的调试。
//
//解决了常规问题，程序现在可以稳定运行，但是结果完全不对，需要自己编写求解K R t