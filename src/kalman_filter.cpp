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

  MatrixXd F_T;
  
  // predict the state
  F_T = F_.transpose();
  x_ = F_ * x_;
  P_ = F_ * P_ * F_T + Q_;
  
  return;
};

void KalmanFilter::Update(const VectorXd &z) {
  
  long x_size;
  VectorXd z_prime, y;
  MatrixXd H_T, S, S_i, K, I;
  
  // update the state using Kalman Filter equations
  z_prime = H_ * x_;
  y = z - z_prime;
  H_T = H_.transpose();
  S = H_ * P_ * H_T + R_;
  S_i = S.inverse();
  K = P_ * H_T * S_i;
  
  // estimate
  x_ = x_ + K * y;
  x_size = x_.size();
  I = MatrixXd::Identity(x_size, x_size);
  P_ = I - K * H_ * P_;
  
  return;
};

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  
  float rho, phi, rho_dot;
  long x_size;
  VectorXd z_prime(3), y;
  MatrixXd H_T, S, S_i, K, I;
  
  // update the state using Extended Kalman Filter equations
  rho = pow(pow(x_(0), 2) + pow(x_(1), 2), 0.5);
  phi = atan2(x_(1), x_(0));
  if (fabs(rho) < 0.0001){
    rho_dot = 0;
  }
  else {
    rho_dot = (x_(0) * x_(2) + x_(1) * x_(3)) / rho;
  };
  z_prime << rho, phi, rho_dot;
  y = z - z_prime;
  H_T = H_.transpose();
  S = H_ * P_ * H_T + R_;
  S_i = S.inverse();
  K = P_ * H_T * S_i;
  
  // estimate
  x_ = x_ + K * y;
  x_size = x_.size();
  I = MatrixXd::Identity(x_size, x_size);
  P_ = I - K * H_ * P_;
  
  return;
};
