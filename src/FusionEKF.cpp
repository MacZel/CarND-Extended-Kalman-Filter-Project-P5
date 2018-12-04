#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);
  ekf_.F_ = MatrixXd::Identity(4, 4);
  ekf_.P_ = MatrixXd::Identity(4, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0,      0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0,      0,
              0,    0.0009, 0,
              0,    0,      0.09;
  
  //measurement matrix - laser
  H_laser_ << 1, 0, 0, 0,
              0, 1, 1, 1;
  
  //initial jacobian
  Hj_ << 1, 1, 0, 0,
         1, 1, 0, 0,
         1, 1, 1, 1;
  
  
};

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {

  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {

    // first measurement
    cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;
    
    // initialize the state - Radar
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    
      // rho - range
      // phi - bearing
      // rho_dot - radial_velocity
      float rho, phi, rho_dot;
      
      rho = measurement_pack.raw_measurements_(0);
      phi = measurement_pack.raw_measurements_(1);
      rho_dot = measurement_pack.raw_measurements_(2);
      
      ekf_.x_ << rho * cos(phi),
                rho * sin(phi),
                rho_dot * cos(phi),
                rho_dot * sin(phi);
    }
    // initialize the state - Laser
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      ekf_.x_ << measurement_pack.raw_measurements_(0),
                measurement_pack.raw_measurements_(1),
                0, 0;
    };
    
    // initialize timestamp
    previous_timestamp_ = measurement_pack.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    
    return;
  };

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  float dt, dt_prim, dt_bis, dt_cer, noise_ax, noise_ay;
  
  // Update the state transition matrix F according to the new elapsed time.
  // - Time is measured in seconds.
  // Update the process noise covariance matrix.
  noise_ax = 9;
  noise_ay = 9;
  
  // multiplied by 10^(-6) to get nanoseconds
  dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
  dt_prim = pow(dt, 2);
  dt_bis = dt_prim * dt;
  dt_cer = dt_bis * dt;
  
  ekf_.F_(0, 2) = dt;
  ekf_.F_(1, 3) = dt;
  
  ekf_.Q_ = MatrixXd(4, 4);
  ekf_.Q_ << dt_cer / 4 * noise_ax, 0,                     dt_bis / 2 * noise_ax, 0,
             0,                     dt_cer / 4 * noise_ay, 0,                     dt_bis / 2 * noise_ay,
             dt_bis / 2 * noise_ax, 0,                     dt_prim + noise_ax,    0,
             0,                     dt_bis / 2 * noise_ay, 0,                     dt_prim + noise_ay;
  
  // update timestamp
  previous_timestamp_ = measurement_pack.timestamp_;

  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/
  
  // Use the sensor type to perform the update step.
  // Update the state and covariance matrices.
  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    Tools tools;
    Hj_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.H_ = Hj_;
    ekf_.R_ = R_radar_;
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    // Laser updates
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_radar_;
    ekf_.Update(measurement_pack.raw_measurements_);
  };

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
  
  return;
};
