#ifndef POSE_ESTIMATOR_9KXR642X
#define POSE_ESTIMATOR_9KXR642X

#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace pose_estimator
{

enum PoseEstimateStatus {
 SUCCESS,
 INSUFFICIENT_INLIERS,
 OPTIMIZATION_FAILURE,
 REPROJECTION_ERROR
};

class PoseEstimator {
public:
  PoseEstimator(const Eigen::Matrix<double, 3, 4>& proj_matrix) :
      proj_matrix_(proj_matrix),
      clique_inlier_threshold_(0.2),
      min_inliers_(10),
      reproj_error_threshold_(2.0)
  { }

  PoseEstimateStatus estimate(const Eigen::Matrix4Xd& ref_xyzw,
                              const Eigen::Matrix4Xd& target_xyzw,
                              std::vector<char> * inliers,
                              Eigen::Isometry3d * estimate);

  double clique_inlier_threshold() { return clique_inlier_threshold_; }
  void set_clique_inlier_threshold(double val) { clique_inlier_threshold_ = val; }

  int min_inliers() { return min_inliers_; }
  void min_inliers(int val) { min_inliers_ = val; }

  double reproj_error_threshold() { return reproj_error_threshold_; }
  void set_reproj_error_threshold(double val) { reproj_error_threshold_ = val; }

private:
  // Helper struct to keep track of matches
  struct MatchInfo {
    int ix;
    std::vector<char> compatibility_vec;
    int compatibility_degree;
  };

  static bool compatibility_degree_cmp(const MatchInfo& x,
                                       const MatchInfo& y) {
    return x.compatibility_degree > y.compatibility_degree;
  }

  int detectInliers(const Eigen::Matrix3Xd& ref_xyz,
                    const Eigen::Matrix3Xd& target_xyz,
                    std::vector<char> * inliers);

  Eigen::Matrix<double, 3, 4> proj_matrix_;
  std::vector<int> inlier_indices_;

  double clique_inlier_threshold_;
  int min_inliers_;
  double reproj_error_threshold_;

};

} /*  */

#endif /* end of include guard: POSE_ESTIMATOR_9KXR642X */
