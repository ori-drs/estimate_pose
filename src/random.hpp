#ifndef RANDOM_HE9Y5O2I
#define RANDOM_HE9Y5O2I

#include <cstdlib>
#include <ctime>
#include <cmath>

namespace pose_estimator
{

class Random {
 public:
  // stupid, thread-unsafe random number generation
  static void clock_seed() {
    srand(static_cast<unsigned int>(clock()));
  }

  static double random() {
    return double(rand())/(double(RAND_MAX) + 1.0);
  }

  static double uniform(double a, double b) {
    return (b - a)*random() + a;
  }

  // return int in range [a,b)
  static int random_int(int a, int b) {
    double u = floor(random()*(b - a) + a);
    return static_cast<int>(u);
  }
};

} /*  */

#endif /* end of include guard: RANDOM_HE9Y5O2I */
