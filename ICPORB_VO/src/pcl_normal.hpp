/* -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.


* File Name : pcl_normal.hpp

* Purpose : Calculate normals extract from pcl 

* Creation Date : 2019-07-09

* Last Modified : Thu Aug 20 16:16:19 2020

* Created By :  Ji-Ying, Li

*_._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._.*/
#ifndef PCL_NORMAL_HPP
#define PCL_NORMAL_HPP 

#include <Eigen/Core>

#define PCL_ADD_POINT4D \
  union EIGEN_ALIGN16{ \
    float data[4]; \
    struct { \
      float x; \
      float y; \
      float z; \
    }; \
  } ; \
  inline Eigen::Map<Eigen::Vector3f> getVector3fMap () { return (Eigen::Vector3f::Map (data)); } \
  inline const Eigen::Map<const Eigen::Vector3f> getVector3fMap () const { return (Eigen::Vector3f::Map (data)); } \
  inline Eigen::Map<Eigen::Array3f> getArray3fMap () { return (Eigen::Array3f::Map (data)); } \
  inline const Eigen::Map<const Eigen::Array3f> getArray3fMap () const { return (Eigen::Array3f::Map (data)); } \

struct _PointXYZ
{
  PCL_ADD_POINT4D;  // This adds the members x,y,z which can also be accessed using the point (which is float[4])
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};
struct PointXYZ : public _PointXYZ
{
  inline PointXYZ()
  {
    x = y = z = 0.0f;
    data[3] = 1.0f;
  }
  inline PointXYZ (float _x, float _y, float _z) { x = _x; y = _y; z = _z; data[3] = 1.0f; }
};

template <typename PointT, typename Scalar> inline unsigned int
computeMeanAndCovarianceMatrix (const std::vector<PointT> &cloud,
                                     Eigen::Matrix<Scalar, 3, 3> &covariance_matrix,
                                     Eigen::Matrix<Scalar, 4, 1> &centroid)
{
  // create the buffer on the stack which is much faster than using cloud[indices[i]] and centroid as a buffer
  Eigen::Matrix<Scalar, 1, 9, Eigen::RowMajor> accu = Eigen::Matrix<Scalar, 1, 9, Eigen::RowMajor>::Zero ();
  size_t point_count;
  point_count = cloud.size ();
  // For each point in the cloud
  for (size_t i = 0; i < point_count; ++i)
  {
    accu [0] += cloud[i].x * cloud[i].x;
    accu [1] += cloud[i].x * cloud[i].y;
    accu [2] += cloud[i].x * cloud[i].z;
    accu [3] += cloud[i].y * cloud[i].y; // 4
    accu [4] += cloud[i].y * cloud[i].z; // 5
    accu [5] += cloud[i].z * cloud[i].z; // 8
    accu [6] += cloud[i].x;
    accu [7] += cloud[i].y;
    accu [8] += cloud[i].z;
  }
  
  accu /= static_cast<Scalar> (point_count);
  if (point_count != 0)
  {
    //centroid.head<3> () = accu.tail<3> ();    -- does not compile with Clang 3.0
    centroid[0] = accu[6]; centroid[1] = accu[7]; centroid[2] = accu[8];
    centroid[3] = 1;
    covariance_matrix.coeffRef (0) = accu [0] - accu [6] * accu [6];
    covariance_matrix.coeffRef (1) = accu [1] - accu [6] * accu [7];
    covariance_matrix.coeffRef (2) = accu [2] - accu [6] * accu [8];
    covariance_matrix.coeffRef (4) = accu [3] - accu [7] * accu [7];
    covariance_matrix.coeffRef (5) = accu [4] - accu [7] * accu [8];
    covariance_matrix.coeffRef (8) = accu [5] - accu [8] * accu [8];
    covariance_matrix.coeffRef (3) = covariance_matrix.coeff (1);
    covariance_matrix.coeffRef (6) = covariance_matrix.coeff (2);
    covariance_matrix.coeffRef (7) = covariance_matrix.coeff (5);
  }
  return (static_cast<unsigned int> (point_count));
}
template <typename PointT, typename Scalar> inline void
flipNormalTowardsViewpoint (const PointT &point, float vp_x, float vp_y, float vp_z,
                            Eigen::Matrix<Scalar, 4, 1>& normal)
{
  Eigen::Matrix <Scalar, 4, 1> vp (vp_x - point.x, vp_y - point.y, vp_z - point.z, 0);

  // Dot product between the (viewpoint - point) and the plane normal
  float cos_theta = vp.dot (normal);

  // Flip the plane normal
  if (cos_theta < 0)
  {
    normal *= -1;
    normal[3] = 0.0f;
    // Hessian form (D = nc . p_plane (centroid here) + p)
    normal[3] = -1 * normal.dot (Eigen::Vector4f::Map(point.data));
  }
}
template <typename Scalar, typename Roots> inline void
computeRoots2 (const Scalar& b, const Scalar& c, Roots& roots)
{
  roots (0) = Scalar (0);
  Scalar d = Scalar (b * b - 4.0 * c);
  if (d < 0.0)  // no real roots ! THIS SHOULD NOT HAPPEN!
    d = 0.0;

  Scalar sd = ::std::sqrt (d);

  roots (2) = 0.5f * (b + sd);
  roots (1) = 0.5f * (b - sd);
}
template <typename Matrix, typename Roots> inline void
computeRoots (const Matrix& m, Roots& roots)
{
  typedef typename Matrix::Scalar Scalar;

  // The characteristic equation is x^3 - c2*x^2 + c1*x - c0 = 0.  The
  // eigenvalues are the roots to this equation, all guaranteed to be
  // real-valued, because the matrix is symmetric.
  Scalar c0 =      m (0, 0) * m (1, 1) * m (2, 2)
      + Scalar (2) * m (0, 1) * m (0, 2) * m (1, 2)
             - m (0, 0) * m (1, 2) * m (1, 2)
             - m (1, 1) * m (0, 2) * m (0, 2)
             - m (2, 2) * m (0, 1) * m (0, 1);
  Scalar c1 = m (0, 0) * m (1, 1) -
        m (0, 1) * m (0, 1) +
        m (0, 0) * m (2, 2) -
        m (0, 2) * m (0, 2) +
        m (1, 1) * m (2, 2) -
        m (1, 2) * m (1, 2);
  Scalar c2 = m (0, 0) + m (1, 1) + m (2, 2);

  if (fabs (c0) < Eigen::NumTraits < Scalar > ::epsilon ())  // one root is 0 -> quadratic equation
    computeRoots2 (c2, c1, roots);
  else
  {
    const Scalar s_inv3 = Scalar (1.0 / 3.0);
    const Scalar s_sqrt3 = std::sqrt (Scalar (3.0));
    // Construct the parameters used in classifying the roots of the equation
    // and in solving the equation for the roots in closed form.
    Scalar c2_over_3 = c2 * s_inv3;
    Scalar a_over_3 = (c1 - c2 * c2_over_3) * s_inv3;
    if (a_over_3 > Scalar (0))
      a_over_3 = Scalar (0);

    Scalar half_b = Scalar (0.5) * (c0 + c2_over_3 * (Scalar (2) * c2_over_3 * c2_over_3 - c1));

    Scalar q = half_b * half_b + a_over_3 * a_over_3 * a_over_3;
    if (q > Scalar (0))
      q = Scalar (0);

    // Compute the eigenvalues by solving for the roots of the polynomial.
    Scalar rho = std::sqrt (-a_over_3);
    Scalar theta = std::atan2 (std::sqrt (-q), half_b) * s_inv3;
    Scalar cos_theta = std::cos (theta);
    Scalar sin_theta = std::sin (theta);
    roots (0) = c2_over_3 + Scalar (2) * rho * cos_theta;
    roots (1) = c2_over_3 - rho * (cos_theta + s_sqrt3 * sin_theta);
    roots (2) = c2_over_3 - rho * (cos_theta - s_sqrt3 * sin_theta);

    // Sort in increasing order.
    if (roots (0) >= roots (1))
      std::swap (roots (0), roots (1));
    if (roots (1) >= roots (2))
    {
      std::swap (roots (1), roots (2));
      if (roots (0) >= roots (1))
        std::swap (roots (0), roots (1));
    }

    if (roots (0) <= 0)  // eigenval for symmetric positive semi-definite matrix can not be negative! Set it to 0
      computeRoots2 (c2, c1, roots);
  }
}
template <typename Matrix, typename Vector> inline void
eigen33 (const Matrix& mat, typename Matrix::Scalar& eigenvalue, Vector& eigenvector)
{
  typedef typename Matrix::Scalar Scalar;
  // Scale the matrix so its entries are in [-1,1].  The scaling is applied
  // only when at least one matrix entry has magnitude larger than 1.

  Scalar scale = mat.cwiseAbs ().maxCoeff ();
  if (scale <= std::numeric_limits < Scalar > ::min ())
    scale = Scalar (1.0);

  Matrix scaledMat = mat / scale;

  Vector eigenvalues;
  computeRoots (scaledMat, eigenvalues);

  eigenvalue = eigenvalues (0) * scale;

  scaledMat.diagonal ().array () -= eigenvalues (0);

  Vector vec1 = scaledMat.row (0).cross (scaledMat.row (1));
  Vector vec2 = scaledMat.row (0).cross (scaledMat.row (2));
  Vector vec3 = scaledMat.row (1).cross (scaledMat.row (2));

  Scalar len1 = vec1.squaredNorm ();
  Scalar len2 = vec2.squaredNorm ();
  Scalar len3 = vec3.squaredNorm ();

  if (len1 >= len2 && len1 >= len3)
    eigenvector = vec1 / std::sqrt (len1);
  else if (len2 >= len1 && len2 >= len3)
    eigenvector = vec2 / std::sqrt (len2);
  else
    eigenvector = vec3 / std::sqrt (len3);
}
inline void
solvePlaneParameters (const Eigen::Matrix3f &covariance_matrix,
                           float &nx, float &ny, float &nz, float &curvature)
{
  // Avoid getting hung on Eigen's optimizers
//  for (int i = 0; i < 9; ++i)
//    if (!pcl_isfinite (covariance_matrix.coeff (i)))
//    {
//      //PCL_WARN ("[solvePlaneParameteres] Covariance matrix has NaN/Inf values!\n");
//      nx = ny = nz = curvature = std::numeric_limits<float>::quiet_NaN ();
//      return;
//    }
  // Extract the smallest eigenvalue and its eigenvector
  EIGEN_ALIGN16 Eigen::Vector3f::Scalar eigen_value;
  EIGEN_ALIGN16 Eigen::Vector3f eigen_vector;
  eigen33 (covariance_matrix, eigen_value, eigen_vector);

  nx = eigen_vector [0];
  ny = eigen_vector [1];
  nz = eigen_vector [2];

  // Compute the curvature surface change
  float eig_sum = covariance_matrix.coeff (0) + covariance_matrix.coeff (4) + covariance_matrix.coeff (8);
  if (eig_sum != 0)
    curvature = fabsf (eigen_value / eig_sum);
  else
    curvature = 0;
}
inline void
solvePlaneParameters (const Eigen::Matrix3f &covariance_matrix,
                           const Eigen::Vector4f &point,
                           Eigen::Vector4f &plane_parameters, float &curvature)
{
  solvePlaneParameters (covariance_matrix, plane_parameters [0], plane_parameters [1], plane_parameters [2], curvature);

  plane_parameters[3] = 0;
  // Hessian form (D = nc . p_plane (centroid here) + p)
  plane_parameters[3] = -1 * plane_parameters.dot (point);
}
template <typename PointT> inline bool
  computePointNormal (const std::vector <PointT> &cloud,
                      Eigen::Vector4f &plane_parameters, float &curvature)
  {
    // Placeholder for the 3x3 covariance matrix at each surface patch
    EIGEN_ALIGN16 Eigen::Matrix3f covariance_matrix;
    // 16-bytes aligned placeholder for the XYZ centroid of a surface patch
    Eigen::Vector4f xyz_centroid;
    if (cloud.size () < 3 ||
        computeMeanAndCovarianceMatrix (cloud, covariance_matrix, xyz_centroid) == 0)
    {
      plane_parameters.setConstant (std::numeric_limits<float>::quiet_NaN ());
      curvature = std::numeric_limits<float>::quiet_NaN ();
      return false;
    }
    // Get the plane normal and surface curvature
    solvePlaneParameters (covariance_matrix, xyz_centroid, plane_parameters, curvature);
    return true;
}

#endif /* PCL_NORMAL_HPP */
