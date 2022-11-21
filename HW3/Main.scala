import Linear_regression.linear_regression
import breeze.linalg.{*, DenseMatrix, DenseVector}
import breeze.stats.distributions.Uniform
import breeze.stats.distributions.Rand.FixedSeed.randBasis

object Main {
  def main(args: Array[String]): Unit = {
    val a = new linear_regression(loss = "MAE", reg = "l1")
    val X = DenseMatrix.rand(1000, 55, Uniform(-5000, 5000))
    val y = DenseVector.rand(1000, Uniform(-5000, 5000))
    a.fit(X, y)
    println(a.predict(X))
  }
}