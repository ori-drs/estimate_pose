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
      proj_matrix_(proj_matrix)
  { }

  PoseEstimateStatus estimate(const Eigen::Matrix4Xd& ref_xyzw,
                              const Eigen::Matrix4Xd& target_xyzw,
                              std::vector<char> * inliers,
                              Eigen::Isometry3d * estimate);

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

  int detect_inliers(const Eigen::Matrix3Xd& ref_xyz,
                     const Eigen::Matrix3Xd& target_xyz,
                     std::vector<char> * inliers);

  Eigen::Matrix<double, 3, 4> proj_matrix_;
  std::vector<MatchInfo> matches_;
  std::vector<int> inlier_indices_;

};

} /*  */

#endif /* end of include guard: POSE_ESTIMATOR_9KXR642X */
