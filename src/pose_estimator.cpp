#include "pose_estimator.hpp"

#include <algorithm>
#include <iostream>

#include "refine_motion_estimate.hpp"

namespace pose_estimator
{

int
PoseEstimator::detect_inliers(const Eigen::Matrix3Xd& ref_xyz,
                              const Eigen::Matrix3Xd& target_xyz,
                              std::vector<char> * inliers)
{
  static const double clique_inlier_threshold = 0.2;
  assert (ref_xyz.cols() == target_xyz.cols());
  assert (ref_xyz.cols() > 0);

  inlier_indices_.clear();
  int num_matches = ref_xyz.cols();
  matches_.resize(num_matches);

  for (int i=0; i < num_matches; ++i) {
    matches_[i].ix = i;
    matches_[i].compatibility_vec.resize(num_matches, 0);
    matches_[i].compatibility_vec[i] = 1;
    matches_[i].compatibility_degree = 1;
  }

  for (int i=0; i < num_matches; ++i) {
    const Eigen::Vector3d& ref_xyz_i(ref_xyz.col(i));
    const Eigen::Vector3d& target_xyz_i(target_xyz.col(i));
    for (int j=i+1; j < num_matches; ++j) {
      const Eigen::Vector3d& ref_xyz_j(ref_xyz.col(j));
      const Eigen::Vector3d& target_xyz_j(target_xyz.col(j));
      double ref_dist = (ref_xyz_i-ref_xyz_j).norm();
      double target_dist = (target_xyz_i-target_xyz_j).norm();
      bool consistent = fabs(ref_dist - target_dist) < clique_inlier_threshold;
      if (consistent) {
        matches_[i].compatibility_vec[j] = 1;
        matches_[j].compatibility_vec[i] = 1;
        matches_[i].compatibility_degree++;
        matches_[j].compatibility_degree++;
      }
    }
  }

  // sort the matches based on their compatibility with other matches
  std::sort(matches_.begin(), matches_.end(), compatibility_degree_cmp);

  // assume all matches are inliers
  inliers->clear(); inliers->resize(num_matches, 1);

  // mark first match as inlier and reject everyone incompatible with it
  for (int j=0; j < num_matches; ++j) {
    if (!matches_[0].compatibility_vec[j]) { inliers->at(j) = 0; }
  }
  inlier_indices_.push_back(matches_[0].ix);

  // now start adding inliers that are consistent with all existing
  // inliers
  for (std::vector<MatchInfo>::iterator itr = matches_.begin()+1;
       itr != matches_.end();
       ++itr) {
    // if this candidate is consistent with fewer than the existing number
    // of inliers, then immediately stop iterating since no more matches can
    // be inliers
    if (itr->compatibility_degree < static_cast<int>(inlier_indices_.size())) { break; }
    // skip if we know it's not an inlier
    if (!inliers->at(itr->ix)) { continue; }
    // else add it to clique
    for (int j=0; j < num_matches; ++j) {
      if (!itr->compatibility_vec[j]) { inliers->at(j) = 0; }
    }
    inlier_indices_.push_back(itr->ix);
  }

  return static_cast<int>(inlier_indices_.size());
}


PoseEstimateStatus
PoseEstimator::estimate(const Eigen::Matrix4Xd& ref_xyzw,
                        const Eigen::Matrix4Xd& target_xyzw,
                        std::vector<char> * inliers,
                        Eigen::Isometry3d * estimate)
{
  static const int min_inliers = 10;
  static const double reproj_error_threshold = 2.0;
  using namespace Eigen;

  Matrix3Xd ref_xyz = ref_xyzw.topRows<3>().cwiseQuotient(ref_xyzw.row(3).replicate<3, 1>());
  Matrix3Xd target_xyz = target_xyzw.topRows<3>().cwiseQuotient(target_xyzw.row(3).replicate<3,1>());

  int num_inliers = detect_inliers(ref_xyz, target_xyz, inliers);

  if (num_inliers < min_inliers) {
    return INSUFFICIENT_INLIERS;
  }

  // estimate rigid body transform with inliers
  Matrix4Xd inlier_ref_xyzw(4, num_inliers);
  Matrix4Xd inlier_target_xyzw(4, num_inliers);
  Matrix3Xd inlier_ref_xyz(3, num_inliers);
  Matrix3Xd inlier_target_xyz(3, num_inliers);
  for (int i=0; i < num_inliers; ++i) {
    int ix = inlier_indices_[i];
    inlier_ref_xyzw.col(i) = ref_xyzw.col(ix);
    inlier_target_xyzw.col(i) = target_xyzw.col(ix);
    inlier_ref_xyz.col(i) = ref_xyz.col(ix);
    inlier_target_xyz.col(i) = target_xyz.col(ix);
  }

  Matrix4d ume_estimate = Eigen::umeyama(inlier_target_xyz, inlier_ref_xyz);
  *estimate = Isometry3d(ume_estimate);

  // get inlier projections
  Matrix3Xd inlier_ref_uvw = proj_matrix_ * inlier_ref_xyzw;
  Matrix2Xd inlier_ref_uv = inlier_ref_uvw.topRows<2>().cwiseQuotient(inlier_ref_uvw.row(2).replicate<2, 1>());
  Matrix3Xd inlier_target_uvw = proj_matrix_ * inlier_target_xyzw;
  Matrix2Xd inlier_target_uv = inlier_target_uvw.topRows<2>().cwiseQuotient(inlier_target_uvw.row(2).replicate<2, 1>());

  MatrixXd estimate_covariance;
  // refine bidirectional projection error
  fovis::refineMotionEstimateBidirectional(inlier_ref_xyzw,
                                           inlier_ref_uv,
                                           inlier_target_xyzw,
                                           inlier_target_uv,
                                           proj_matrix_(0, 0),
                                           proj_matrix_(0, 2),
                                           proj_matrix_(1, 2),
                                           *estimate,
                                           6,
                                           estimate,
                                           &estimate_covariance);

  // compute projection error
  Matrix3Xd reproj_inlier_target_uvw = proj_matrix_ * estimate->matrix() * inlier_target_xyzw;
  Matrix2Xd reproj_inlier_target_uv = reproj_inlier_target_uvw.topRows<2>().cwiseQuotient(reproj_inlier_target_uvw.row(2).replicate<2, 1>());
  VectorXd reproj_error = (reproj_inlier_target_uv - inlier_ref_uv).colwise().norm();

  // remove features with high projection error
  std::vector<int> tmp_inlier_indices;
  for (int i=0; i < num_inliers; ++i) {
    if (reproj_error(i) > reproj_error_threshold) {
      inliers->at(inlier_indices_.at(i)) = 0;
    } else {
      tmp_inlier_indices.push_back(inlier_indices_.at(i));
    }
  }
  // see if things have changed. If not, we're done.
  if (tmp_inlier_indices.size()==inlier_indices_.size()) { return SUCCESS; }
  // we removed more outliers, update outliers and re-refine.
  inlier_indices_.swap(tmp_inlier_indices);
  num_inliers = static_cast<int>(inlier_indices_.size());
  if (num_inliers < min_inliers) {
    return INSUFFICIENT_INLIERS;
  }

  inlier_ref_xyzw = Matrix4Xd(4, num_inliers);
  inlier_target_xyzw = Matrix4Xd(4, num_inliers);
  for (int i=0; i < num_inliers; ++i) {
    int ix = inlier_indices_[i];
    inlier_ref_xyzw.col(i) = ref_xyzw.col(ix);
    inlier_target_xyzw.col(i) = target_xyzw.col(ix);
  }
  inlier_ref_uvw = proj_matrix_ * inlier_ref_xyzw;
  inlier_ref_uv = inlier_ref_uvw.topRows<2>().cwiseQuotient(inlier_ref_uvw.row(2).replicate<2, 1>());
  inlier_target_uvw = proj_matrix_ * inlier_target_xyzw;
  inlier_target_uv = inlier_target_uvw.topRows<2>().cwiseQuotient(inlier_target_uvw.row(2).replicate<2, 1>());
  fovis::refineMotionEstimateBidirectional(inlier_ref_xyzw,
                                           inlier_ref_uv,
                                           inlier_target_xyzw,
                                           inlier_target_uv,
                                           proj_matrix_(0, 0),
                                           proj_matrix_(0, 2),
                                           proj_matrix_(1, 2),
                                           *estimate,
                                           6,
                                           estimate,
                                           &estimate_covariance);

  return SUCCESS;

	// generate uvd1 -> xyzw matrix, aka Q or reprojection matrix.
	//Matrix4d reproj;
	//reproj <<
	//		1./camera_(0,0), 0, 0, -camera_(0, 2)/camera_(0, 0),
	//		0, 1./camera_(1, 1), 0, -camera_(1, 2)/camera_(1, 1),
	//		0, 0, 0, 1,
	//		0, 0, -1./(camera_(0, 0)*baseline_), 0;

}

} /*  */
