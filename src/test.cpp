
#include <assert.h>

#include <iostream>
#include <vector>
#include <numeric>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "pose_estimator.hpp"
#include "random.hpp"

#define NUM_MATCHES 400
#define OUTLIER_PROPORTION 0.95

static Eigen::Isometry3d
random_transformation()
{
	double u = pose_estimator::Random::random();
	double r1 = sqrt(1.0 - u);
	double r2 = sqrt(u);
	double t1 = pose_estimator::Random::uniform(0., 2.*M_PI);
	double t2 = pose_estimator::Random::uniform(0., 2.*M_PI);
	//wxyz
	Eigen::Quaterniond quat(cos(t2)*r2, sin(t1)*r1, cos(t1)*r1, sin(t2)*r2);
	Eigen::Isometry3d tf; tf.setIdentity();
	tf.translate(10*Eigen::Vector3d::Random());
	tf.rotate(quat);
	return tf;
}

int main(int argc, char *argv[]) {
  using namespace Eigen;
  using namespace pose_estimator;

	pose_estimator::Random::clock_seed();

	for (int iter=0; iter < 10; ++iter) {
		// random points
		Matrix4Xd dst_xyz1 = Matrix4Xd::Ones(4, NUM_MATCHES);
		dst_xyz1.topRows<3>() = 10*Matrix3Xd::Random(3, NUM_MATCHES);

		Isometry3d tf = random_transformation();
		Matrix4Xd src_xyz1 = tf*dst_xyz1;

		// add some uniform noise to make it more interesting
		//src_xyz1.topRows<3>() += 1e-3*Matrix3Xd::Random(3, NUM_MATCHES);

		// make the first matches 'outliers' by shuffling.
		int num_outliers = static_cast<int>(OUTLIER_PROPORTION*NUM_MATCHES);
		for (int i=0; i < num_outliers; ++i) {
			int j = pose_estimator::Random::random_int(i, num_outliers);
			src_xyz1.col(i).swap(src_xyz1.col(j));
		}

		Matrix<double, 3, 4> proj;
		proj <<   568 , 0   , 377, 0,
							0   , 568 , 240, 0,
							0   , 0   , 1  , 0;

		PoseEstimator pe(proj);

		std::vector<char> inliers;
		Isometry3d estimate;
		int status = pe.estimate(src_xyz1, dst_xyz1, &inliers, &estimate);

		int num_inliers = std::accumulate(inliers.begin(), inliers.end(), 0);
		// pick out inliers
		Matrix4Xd inlier_src_xyz(4, num_inliers);
		Matrix4Xd inlier_dst_xyz(4, num_inliers);
		for (size_t ii=0, j=0; ii<inliers.size(); ++ii) {
			if (inliers.at(ii)) {
				inlier_src_xyz.col(j) = src_xyz1.col(ii);
				inlier_dst_xyz.col(j) = dst_xyz1.col(ii);
				++j;
			}
		}

		Matrix4Xd est_inlier_src_xyz = estimate * inlier_dst_xyz;
		double mean_err = (est_inlier_src_xyz - inlier_src_xyz).colwise().norm().mean();

		std::cerr << "mean inlier 3D err = " << mean_err << std::endl;
		std::cerr << "\n";
		std::cerr << "tf = \n";
		std::cerr << tf.translation() << "\n";
		std::cerr << tf.rotation() << "\n";
		std::cerr << "\n";
		std::cerr << "estimate = \n";
		std::cerr << estimate.translation() << "\n";
		std::cerr << estimate.rotation() << "\n";
		std::cerr << "\n";
		std::cerr << "inliers = \n";
		for (size_t i=0; i<inliers.size(); ++i) {
			std::cerr << static_cast<int>(inliers[i]);
		}
    bool is_approx = estimate.isApprox(tf);
    std::cerr << "\nApproximately equal: " << (is_approx? "True" : "False") << "\n";
		std::cerr << "\n***\n\n";
	}

  return 0;
}
