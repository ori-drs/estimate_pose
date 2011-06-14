#include "pose_estimator.hpp"

#include <algorithm>
#include <iostream>

#include <fovis/refine_motion_estimate.hpp>

#include "random.hpp"

namespace pose_estimator
{

const char *PoseEstimateStatusStrings[] = {
  "SUCCESS",
  "INSUFFICIENT_INLIERS",
  "OPTIMIZATION_FAILURE",
  "REPROJECTION_ERROR"
};

int PoseEstimator::detectInliersClique(const Eigen::Matrix3Xd& ref_xyz,
                                       const Eigen::Matrix3Xd& target_xyz,
                                       const Eigen::Matrix4Xd& ref_xyzw,
                                       const Eigen::Matrix4Xd& target_xyzw,
                                       std::vector<char> * inliers) {
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
    const Eigen::Vector4d& ref_xyzw_i(ref_xyzw.col(i));
    const Eigen::Vector4d& target_xyzw_i(target_xyzw.col(i));

    // are the features points at infinity?
    bool ref_infinity = ref_xyzw_i.w() < 1e-9;
    bool target_infinity = target_xyzw_i.w() < 1e-9;

    for (int j=i+1; j < num_matches; ++j) {
      const Eigen::Vector3d& ref_xyz_j(ref_xyz.col(j));
      const Eigen::Vector3d& target_xyz_j(target_xyz.col(j));
      const Eigen::Vector4d& ref_xyzw_j(ref_xyzw.col(j));
      const Eigen::Vector4d& target_xyzw_j(target_xyzw.col(j));

      bool compatible;
      // special case:  if either of the features are points at infinity, then
      // we can't compare their distances.
      if((ref_infinity && ref_xyzw_j.w() < 1e-9) ||
         (target_infinity && target_xyzw_j.w() < 1e-9)) {
        compatible = true;
      } else {
        // just use 3D euclidean distance
        double ref_dist = (ref_xyz_i-ref_xyz_j).norm();
        double target_dist = (target_xyz_i-target_xyz_j).norm();
        compatible = fabs(ref_dist - target_dist) < clique_inlier_threshold_;
      }
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
    if (!matches[0].compatibility_vec[j]) { (*inliers)[j] = 0; }
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
    if (!(*inliers)[itr->ix]) { continue; }
    // else add it to clique
    for (int j=0; j < num_matches; ++j) {
      if (!itr->compatibility_vec[j]) { (*inliers)[j] = 0; }
    }
    inlier_indices_.push_back(itr->ix);
  }

  return static_cast<int>(inlier_indices_.size());
}


#ifdef USE_RANSAC
int PoseEstimator::detectInliersRansac(const Eigen::Matrix3Xd& ref_xyz,
                                       const Eigen::Matrix3Xd& target_xyz,
                                       const Eigen::Matrix4Xd& ref_xyzw,
                                       const Eigen::Matrix4Xd& target_xyzw,
                                       std::vector<char> * inliers) {

  if (ref_xyz.cols() < min_inliers_) {
    inlier_indices_.clear();
    return 0;
  }

  // TODO avoid this repetition
  Eigen::Matrix3Xd ref_uvw = proj_matrix_ * ref_xyzw;
  Eigen::Matrix2Xd ref_uv = ref_uvw.topRows<2>().cwiseQuotient(ref_uvw.row(2).replicate<2,1>());

  double min_loss = 1.0e15;
  Eigen::Isometry3d best_estimate;

  //
  // In theory, for .99 probability of finding correct hypothesis
  // according to fraction of outliers
  // 0.50 -> 35
  // 0.55 -> 49
  // 0.60 -> 70
  // 0.65 -> 106
  // 0.70 -> 169
  // 0.75 -> 293
  // 0.80 -> 574
  // 0.85 -> 1363
  // 0.90 -> 4603
  // 0.95 -> 36840
  //

  Eigen::Matrix3d sampled_ref_xyz, sampled_target_xyz;
  int num_matches = ref_xyz.cols();
  for (int i=0; i < sac_iterations_; ++i) {
    // Randomly select 3 different matches
    int a = Random::random_int(0, num_matches);
    int b = a;
    while (a == b) {
      b = Random::random_int(0, num_matches);
    }
    int c = a;
    while (c == a || c == b) {
      c = Random::random_int(0, num_matches);
    }

    sampled_ref_xyz.col(0) = ref_xyz.col(a);
    sampled_ref_xyz.col(1) = ref_xyz.col(b);
    sampled_ref_xyz.col(2) = ref_xyz.col(c);

    sampled_target_xyz.col(0) = target_xyz.col(a);
    sampled_target_xyz.col(1) = target_xyz.col(b);
    sampled_target_xyz.col(2) = target_xyz.col(c);

#if USE_HORN
    Eigen::Isometry3d estimate;
    int ret = absolute_orientation_horn(sampled_target_xyz, sampled_ref_xyz, &estimate);
    if (ret) { continue; }
#else
    Eigen::Isometry3d estimate = Eigen::Isometry3d(Eigen::umeyama(sampled_target_xyz, sampled_ref_xyz));
#endif

    // find inliers based on reprojection
    //bot_tictoc("match scoring");
    Eigen::Matrix3Xd reproj_uvw = proj_matrix_ * estimate.matrix() * target_xyzw;
    Eigen::Matrix2Xd reproj_uv = reproj_uvw.topRows<2>().cwiseQuotient(reproj_uvw.row(2).replicate<2,1>());
    Eigen::VectorXd reproj_err = (reproj_uv - ref_uv).colwise().norm();

    double loss=0;
    for (int m_ind = 0; m_ind < num_matches; ++m_ind) {
      if (m_ind == a || m_ind == b || m_ind == c) { continue; }
      if (reproj_err(m_ind) > reproj_error_threshold_) {
        loss += reproj_error_threshold_;
      } // else do nothing
      if (loss > min_loss) {
        break; // loss always increases so this is not the best
      }
    }
    if (loss < min_loss) {
      min_loss = loss;
      best_estimate = estimate;
    }
    //bot_tictoc("match scoring");
    //printf("ransac %d: inliers: %d\n", i, inliers);
  }

  // mark final inliers
  Eigen::Matrix3Xd reproj_uvw = proj_matrix_ * best_estimate.matrix() * target_xyzw;
  Eigen::Matrix2Xd reproj_uv = reproj_uvw.topRows<2>().cwiseQuotient(reproj_uvw.row(2).replicate<2,1>());
  Eigen::VectorXd reproj_err = (reproj_uv - ref_uv).colwise().norm();

  inlier_indices_.clear();
  inliers->clear(); inliers->resize(num_matches, 0);
  double loss=0;
  for (int m_ind = 0; m_ind < num_matches; ++m_ind) {
    if (reproj_err(m_ind) < reproj_error_threshold_) {
      inlier_indices_.push_back(m_ind);
      (*inliers)[m_ind] = 1;
    }
  }
  int num_inliers = static_cast<int>(inlier_indices_.size());

  return num_inliers;
  // TODO fix redundancy after this
}
#endif

PoseEstimateStatus PoseEstimator::estimate(const Eigen::Matrix4Xd& ref_xyzw,
                                           const Eigen::Matrix4Xd& target_xyzw,
                                           std::vector<char> * inliers,
                                           Eigen::Isometry3d * estimate,
                                           Eigen::MatrixXd * estimate_covariance) {

  Eigen::Matrix3Xd ref_xyz = ref_xyzw.topRows<3>().cwiseQuotient(ref_xyzw.row(3).replicate<3, 1>());
  Eigen::Matrix3Xd target_xyz = target_xyzw.topRows<3>().cwiseQuotient(target_xyzw.row(3).replicate<3,1>());

  int num_matches = ref_xyz.cols();
#ifdef USE_RANSAC
  int num_inliers = detectInliersRansac(ref_xyz, target_xyz, ref_xyzw, target_xyzw, inliers);
#else
  int num_inliers = detectInliersClique(ref_xyz, target_xyz, ref_xyzw, target_xyzw, inliers);
#endif

  if (num_inliers < min_inliers_) {
    if (verbose_) {
      std::cerr << "pose_estimator: Insufficient inliers after max clique" << std::endl;
    }
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
                                           estimate_covariance);

  // compute projection error
  Eigen::Matrix3Xd reproj_target_uvw = proj_matrix_ * estimate->matrix() * target_xyzw;
  Eigen::Matrix2Xd reproj_target_uv = reproj_target_uvw.topRows<2>().cwiseQuotient(reproj_target_uvw.row(2).replicate<2, 1>());
  Eigen::VectorXd reproj_error = (reproj_target_uv - ref_uv).colwise().norm();

  // - remove features with high projection error
  // - recover features with low projection error
  std::vector<int> tmp_inlier_indices;
  for (size_t i=0; i < num_matches; ++i) {
    if (reproj_error(i) < reproj_error_threshold_) {
      (*inliers)[i] = 1;
      tmp_inlier_indices.push_back(i);
    } else {
      (*inliers)[i] = 0;
    }
  }

  // see if things have changed. If not, we're done.
  if (tmp_inlier_indices == inlier_indices_) { return SUCCESS; }
  // we removed more outliers, update outliers and re-refine.
  inlier_indices_.swap(tmp_inlier_indices);
  num_inliers = static_cast<int>(inlier_indices_.size());
  if (num_inliers < min_inliers_) {
    if (verbose_) {
      std::cerr << "pose_estimator: Insufficient inliers after refinement" << std::endl;
    }
    return INSUFFICIENT_INLIERS;
  }

  // again, group remaining inliers
  inlier_ref_xyzw.resize(4, num_inliers);
  inlier_target_xyzw.resize(4, num_inliers);
  inlier_ref_uv.resize(2, num_inliers);
  inlier_target_uv.resize(2, num_inliers);
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
                                           estimate_covariance);

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
