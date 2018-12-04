#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  
  VectorXd rmse(4);
  rmse << 0,0,0,0;
  
  // the estimation vector size should not be zero
  if (estimations.size() == 0) {
    cout << "CalculateRMSE: The estimation vector size should not be zero." << endl;
    return rmse;
  };
  
  // the estimation vector size should equal ground truth vector size
  if (estimations.size() != ground_truth.size()){
    cout << "CalculateRMSE: The estimation vector size should equal ground truth vector size." << endl;
    return rmse;
  };
  
  //accumulate squared residuals
  for (unsigned int i=0; i < estimations.size(); ++i){
    VectorXd residual = estimations[i] - ground_truth[i];
    
    //coefficient-wise multiplication
    residual = residual.array()*residual.array();
    rmse += residual;
  };
  
  //calculate the mean
  rmse = rmse/estimations.size();

  //calculate the squared root
  rmse = rmse.array().sqrt();

  return rmse;
};

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  
  MatrixXd Hj(3,4);
  float px, py, vx, vy,
        sum_of_squares, square_root, one_and_half_root, diff;
  
  //recover state parameters
  px = x_state(0);
  py = x_state(1);
  vx = x_state(2);
  vy = x_state(3);
  
  // px^2 + py^2
  sum_of_squares = pow(px, 2) + pow(py, 2);
  
  //check division by zero
  if (fabs(sum_of_squares) < 0.0001){
    cout << "CalculateJacobian: Zero Division Error" << endl;
    return Hj;
  };
  
  // (px^2 + py^2)^(1/2)
  square_root = pow(sum_of_squares, 0.5);
  
  // (px^2 + py^2)*(px^2 + py^2)^(1/2) = (px^2 + py^2)^(3/2)
  one_and_half_root = sum_of_squares * square_root;
  
  // vx*py - vy*px
  diff = vx*py - vy*px;
  
  //compute the Jacobian matrix
  Hj << px/square_root,            py/square_root,               0,              0,
        -py/sum_of_squares,        px/sum_of_squares,            0,              0,
        py*diff/one_and_half_root, px*(-diff)/one_and_half_root, px/square_root, py/square_root;
  
  return Hj;
};
