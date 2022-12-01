package Linear_regression

import breeze.linalg._
import breeze.numerics._
import breeze.stats.distributions.Rand.FixedSeed.randBasis
import breeze.stats.distributions.Uniform
import breeze.stats.mean

class linear_regression(loss: String = "MSE", reg: String = "none", exec_sol: Boolean = false, val lr: Double = 1e-3,
                        val alph: Double = 1, val max_iter: Int = 300, val batch_sz: Int = 30) {
  private var W: DenseVector[Double] = DenseVector.zeros[Double](1)

  def MSE(X: DenseMatrix[Double], y: DenseVector[Double]): Double = {
    mean(pow(X * W - y, 2))
  }

  def MAE(X: DenseMatrix[Double], y: DenseVector[Double]): Double = {
    mean(abs(X * W - y))
  }

  def l1_reg(): Double = {
    sum(abs(W))
  }

  def l2_reg(): Double = {
    W.t * W
  }

  def mse_derivative(X: DenseMatrix[Double], y: DenseVector[Double]): DenseVector[Double] = {
    2.0 * X.t * (X * W - y) / y.length.toDouble
  }

  def mae_derivative(X: DenseMatrix[Double], y: DenseVector[Double]): DenseVector[Double] = {
    (signum(X * W - y).t * X).t / y.length.toDouble
  }

  def l1_reg_derivative(): DenseVector[Double] = {
    signum(W)
  }

  def l2_reg_derivative(): DenseVector[Double] = {
    2.0 * W
  }

  def fit(X: DenseMatrix[Double], y: DenseVector[Double], verbose: Boolean = true): Unit = {
    assert(Array("MSE", "MAE") contains (loss))
    assert(Array("none", "l1", "l2") contains (reg))
    W = DenseVector.rand(X.cols, Uniform(0, 1))
    if (loss == "MSE" && reg != "l1" && exec_sol) {
      if (reg == "l2")
        W = inv(X.t * X + alph * DenseMatrix.eye[Double](X.cols)) * X.t * y
      else
        W = inv(X.t * X) * X.t * y
      return
    }
    var l = 0;
    var r = min(batch_sz, X.rows)
    for (i <- 1 to max_iter) {
      var d_loss = DenseVector.zeros[Double](y.length);
      var d_reg = DenseVector.zeros[Double](y.length)
      var val_loss = 0.0;
      var val_reg = 0.0
      val X_b = X(l until r, ::).copy // - ?
      val y_b = y(l until r).copy
      if (loss == "MSE") {
        val_loss = MSE(X_b, y_b)
        d_loss = mse_derivative(X_b, y_b)
      }
      else {
        val_loss = MAE(X_b, y_b)
        d_loss = mae_derivative(X_b, y_b)
      }
      if (reg == "l1") {
        val_reg = l1_reg()
        d_reg = l1_reg_derivative()
      }
      else if (reg == "l2") {
        val_reg = l2_reg()
        d_reg = l2_reg_derivative()
      }
      W -= lr * (d_loss + d_reg)
      if (verbose)
        println(s"Iteration ${i}: ${loss} = ${val_loss + val_reg}")
      l += batch_sz
      if (l >= X.rows)
        l = 0
      r = min(l + batch_sz, X.rows)
    }
  }

  def predict(X: DenseMatrix[Double]): DenseVector[Double] = {
    return X * W
  }


}
