#include <iostream>
#include <vector>
#include <cmath>
#include <cfloat>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/SVD>
using namespace std;

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

    // 变量都是与课程PPT相对应的
    Eigen::MatrixXf P;
    Eigen::MatrixXf M;
    Eigen::MatrixXf A;
    Eigen::MatrixXf b;
    Eigen::MatrixXf K;
    Eigen::MatrixXf R;
    Eigen::MatrixXf t;
};

void SingleCamera::composeP() {
    //homework1:  根据输入的二维点和三维点，构造P矩阵

    for (int i = 0; i < point_num; i++) {
        double X = world_coor(i, 0);
        double Y = world_coor(i, 1);
        double Z = world_coor(i, 2);
        double x = pixel_coor(i, 0);
        double y = pixel_coor(i, 1);
  
        P(i, 0) = X;
        P(i, 1) = Y;
        P(i, 2) = Z;   
        P(i, 3) = 1;
        P(i, 4) = 0;
        P(i, 5) = 0;
        P(i, 6) = 0;
        P(i, 7) = 0;
        P(i, 8) = -x * X;
        P(i, 9) = -x * Y;
        P(i, 10) = -x * Z;
        P(i, 11) = -x;

        P(point_num + i, 0) = 0;
        P(point_num + i, 1) = 0;
        P(point_num + i, 2) = 0;
        P(point_num + i, 3) = 0;
        P(point_num + i, 4) = X;
        P(point_num + i, 5) = Y;
        P(point_num + i, 6) = Z;
        P(point_num + i, 7) = 1;
        P(point_num + i, 8) = -y * X;
        P(point_num + i, 9) = -y * Y;
        P(point_num + i, 10) = -y * Z;
        P(point_num + i, 11) = -y;
    }
    cout<<"P:\n"<<P<<endl;
}

void SingleCamera::svdP() {
    //  homework2: 根据P矩阵求解M矩阵和A、b矩阵
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(P, Eigen::ComputeFullV);
    Eigen::MatrixXf V = svd.matrixV();
    Eigen::VectorXf pm = V.col(11);
    for (int i = 0; i < 12; ++i) {
          M(i / 4, i % 4) = pm[i];
    }
    A = M.block(0, 0, 3, 3);
    b = M.block(0, 3, 3, 1); 
    cout<<"M:\n"<<M<<endl;
    cout<<"A:\n"<<A<<endl;
    cout<<"b:\n"<<b.transpose()<<endl;
}

void SingleCamera::workIntrinsicAndExtrinsic() {
    // homeworks3:求解相机的内参和外参
    float zita=0;
    float c_x=0;float c_y=0;
    Eigen::Vector3f a_1=A.row(0);
    Eigen::Vector3f a_2=A.row(1);
    Eigen::Vector3f a_3=A.row(2);
    zita=1.0/a_3.norm();

    c_x = pow(zita, 2) * (a_1*a_3.transpose()).norm();
    c_y = pow(zita, 2) * (a_2*a_3.transpose()).norm();
    float cos_aerfa=-((a_1.cross(a_3))*(a_2.cross(a_3)).transpose()).norm()/(((a_1.cross(a_3)).norm())*((a_2.cross(a_3)).norm())); 
    float aerfa=acos(aerfa);
    float aerfa_1=pow(zita,2)*(a_1.cross(a_3)).norm()*sin(aerfa);
    float aerfa_2=pow(zita,2)*(a_2.cross(a_3)).norm()*sin(aerfa);

    K(0,0)=aerfa_1;
    K(0,1)=-aerfa_1*(1/tan(aerfa));
    K(0,2)=c_x;
    K(1,2)=c_y;
    K(1,1)=aerfa_2/sin(aerfa);
    K(2,2)=1;

    Eigen::Vector3f r1=a_2.cross(a_3)/(a_2.cross(a_3)).norm();
    Eigen::Vector3f r3=a_3/a_3.norm();
    Eigen::Vector3f r2=r3.cross(r1);

    R.row(0)=r1;
    R.row(1)=r2;
    R.row(2)=r3; 

    t=zita*K.inverse()*b;
    cout<<"K: \n"<<K<<endl;
    cout<<"R: \n"<<R<<endl;
    cout<<"t: \n"<<t.transpose()<<endl;

}

void SingleCamera::selfcheck(const Eigen::MatrixXf& w_check, const Eigen::MatrixXf& c_check) {
    float average_err = DBL_MAX;
    // homeworks4:根据homework3求解得到的相机的参数，使用测试点进行验证，计算误差
    float total_err = 0.0;
    for (int i = 0; i < w_check.rows(); i++) {
        Eigen::Vector4f wp = w_check.row(i);
        Eigen::Vector3f cp = M*wp;
        cp /=cp(2);
        Eigen::Vector2f cp_normalized = cp.head(2);
        float err = (cp_normalized - (Eigen::Vector2f)(c_check.row(i))).norm();
        total_err += err;
    }
    average_err = total_err / w_check.rows();
    std::cout << "The average error is " << average_err << "," << std::endl;
    if (average_err > 0.1) {
        std::cout << "which is more than 0.1" << std::endl;
    } else {
        std::cout << "which is smaller than 0.1, the M is acceptable" << std::endl;
    }
}

int main(int argc, char** argv) {

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