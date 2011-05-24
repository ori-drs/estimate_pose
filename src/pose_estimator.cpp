#include "pose_estimator.hpp"

#include <algorithm>
#include <iostream>

#include "refine_motion_estimate.hpp"

namespace pose_estimator
{

int
PoseEstimator::detectInliers(const Eigen::Matrix3Xd& ref_xyz,
                             const Eigen::Matrix3Xd& target_xyz,
                             std::vector<char> * inliers)
{
  assert (ref_xyz.cols() == target_xyz.cols());
  assert (ref_xyz.cols() > 0);

  inlier_indices_.clear();
  int num_matches = ref_xyz.cols();

  std::vector<MatchInfo> matches(num_matches);
  for (int i=0; i < num_matches; ++i) {
    matches[i].ix = i;
    matches[i].compatibility_vec.resize(num_matches, 0);
    matches[i].compatibility_vec[i] = 1;
    matches[i].compatibility_degree = 1;
  }

  // assess compatibility of each pair of matches
  for (int i=0; i < num_matches; ++i) {
    const Eigen::Vector3d& ref_xyz_i(ref_xyz.col(i));
    const Eigen::Vector3d& target_xyz_i(target_xyz.col(i));
    for (int j=i+1; j < num_matches; ++j) {
      const Eigen::Vector3d& ref_xyz_j(ref_xyz.col(j));
      const Eigen::Vector3d& target_xyz_j(target_xyz.col(j));
      double ref_dist = (ref_xyz_i-ref_xyz_j).norm();
      double target_dist = (target_xyz_i-target_xyz_j).norm();
      // just use 3D euclidean distance for now
      bool compatible = fabs(ref_dist - target_dist) < clique_inlier_threshold_;
      if (compatible) {
        matches[i].compatibility_vec[j] = 1;
        matches[j].compatibility_vec[i] = 1;
        matches[i].compatibility_degree++;
        matches[j].compatibility_degree++;
      }
    }
  }

  // sort the matches based on their compatibility with other matches
  std::sort(matches.begin(), matches.end(), compatibility_degree_cmp);

  // assume all matches are inliers
  inliers->clear(); inliers->resize(num_matches, 1);

  // mark first match as inlier and reject everyone incompatible with it
  for (int j=0; j < num_matches; ++j) {
    if (!matches[0].compatibility_vec[j]) { inliers->at(j) = 0; }
  }
  inlier_indices_.push_back(matches[0].ix);

  // now start adding inliers that are consistent with all existing
  // inliers
  for (std::vector<MatchInfo>::iterator itr = matches.begin()+1;
       itr != matches.end();
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
  Eigen::Matrix3Xd ref_xyz = ref_xyzw.topRows<3>().cwiseQuotient(ref_xyzw.row(3).replicate<3, 1>());
  Eigen::Matrix3Xd target_xyz = target_xyzw.topRows<3>().cwiseQuotient(target_xyzw.row(3).replicate<3,1>());

  int num_matches = ref_xyz.cols();
  int num_inliers = detectInliers(ref_xyz, target_xyz, inliers);

  if (num_inliers < min_inliers_) {
    return INSUFFICIENT_INLIERS;
  }

  // estimate rigid body transform with inliers
  Eigen::Matrix4Xd inlier_ref_xyzw(4, num_inliers);
  Eigen::Matrix4Xd inlier_target_xyzw(4, num_inliers);
  Eigen::Matrix3Xd inlier_ref_xyz(3, num_inliers);
  Eigen::Matrix3Xd inlier_target_xyz(3, num_inliers);
  for (int i=0; i < num_inliers; ++i) {
    int ix = inlier_indices_[i];
    inlier_ref_xyzw.col(i) = ref_xyzw.col(ix);
    inlier_target_xyzw.col(i) = target_xyzw.col(ix);
    inlier_ref_xyz.col(i) = ref_xyz.col(ix);
    inlier_target_xyz.col(i) = target_xyz.col(ix);
  }

  Eigen::Matrix4d ume_estimate = Eigen::umeyama(inlier_target_xyz, inlier_ref_xyz);
  *estimate = Eigen::Isometry3d(ume_estimate);

  // get projections
  Eigen::Matrix3Xd ref_uvw = proj_matrix_ * ref_xyzw;
  Eigen::Matrix2Xd ref_uv = ref_uvw.topRows<2>().cwiseQuotient(ref_uvw.row(2).replicate<2,1>());
  Eigen::Matrix3Xd target_uvw = proj_matrix_ * target_xyzw;
  Eigen::Matrix2Xd target_uv = target_uvw.topRows<2>().cwiseQuotient(target_uvw.row(2).replicate<2,1>());

  // select inlier projections
  Eigen::Matrix2Xd inlier_ref_uv(2, num_inliers);
  Eigen::Matrix2Xd inlier_target_uv(2, num_inliers);
  for (int i=0; i < num_inliers; ++i) {
    inlier_ref_uv.col(i) = ref_uv.col(inlier_indices_[i]);
    inlier_target_uv.col(i) = target_uv.col(inlier_indices_[i]);
  }

  // TODO return covariance
  Eigen::MatrixXd estimate_covariance;
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
  Eigen::Matrix3Xd reproj_target_uvw = proj_matrix_ * estimate->matrix() * target_xyzw;
  Eigen::Matrix2Xd reproj_target_uv = reproj_target_uvw.topRows<2>().cwiseQuotient(reproj_target_uvw.row(2).replicate<2, 1>());
  Eigen::VectorXd reproj_error = (reproj_target_uv - ref_uv).colwise().norm();

  // - remove features with high projection error
  // - recover features with low projection error
  std::vector<int> tmp_inlier_indices;
  for (size_t i=0; i < num_matches; ++i) {
    if (reproj_error(i) < reproj_error_threshold_) {
      inliers->at(i) = 1;
      tmp_inlier_indices.push_back(i);
    } else {
      inliers->at(i) = 0;
    }
  }
  // see if things have changed. If not, we're done.
  if (tmp_inlier_indices.size()==inlier_indices_.size()) { return SUCCESS; }
  // we removed more outliers, update outliers and re-refine.
  inlier_indices_.swap(tmp_inlier_indices);
  num_inliers = static_cast<int>(inlier_indices_.size());
  if (num_inliers < min_inliers_) {
    return INSUFFICIENT_INLIERS;
  }

  // again, group remaining inliers
  inlier_ref_xyzw = Eigen::Matrix4Xd(4, num_inliers);
  inlier_target_xyzw = Eigen::Matrix4Xd(4, num_inliers);
  for (int i=0; i < num_inliers; ++i) {
    int ix = inlier_indices_[i];
    inlier_ref_xyzw.col(i) = ref_xyzw.col(ix);
    inlier_target_xyzw.col(i) = target_xyzw.col(ix);
    inlier_ref_uv.col(i) = ref_uv.col(ix);
    inlier_target_uv.col(i) = target_uv.col(ix);
  }

  // final refinement
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
