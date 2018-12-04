#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
  
  return;
};

void KalmanFilter::Predict() {
  
  // predict the state
  MatrixXd F_T = F_.transpose();
  x_ = F_ * x_;
  P_ = F_ * P_ * F_T + Q_;
  
  return;
};

void KalmanFilter::Estimate(const VectorXd &y) {
  
  MatrixXd H_T = H_.transpose();
  MatrixXd S = H_ * P_ * H_T + R_;
  MatrixXd S_i = S.inverse();
  MatrixXd K = P_ * H_T * S_i;
  
  x_ = x_ + (K * y);
  int x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
  
  return;
};

void KalmanFilter::Update(const VectorXd &z) {
  
  VectorXd y = z - H_ * x_;
  Estimate(y);
  
  return;
};

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  
  double px = x_(0);
  double py = x_(1);
  double vx = x_(2);
  double vy = x_(3);
  
  // update the state using Extended Kalman Filter equations
  double rho = sqrt(px*px + py*py);
  double phi = atan(py / px);
  double rho_dot = (px*vx + py*vy) / rho;
  VectorXd z_prime = VectorXd(3);
  z_prime << rho, phi, rho_dot;
  while (z_prime(1) - z(1) > M_PI/2) {
    z_prime(1) -= M_PI;
  };
  while (z(1) - z_prime(1) > M_PI/2) {
    z_prime(1) += M_PI;
  };
  VectorXd y = z - z_prime;
  Estimate(y);
  
  return;
};
