package org.apache.spark.ml.made

import breeze.stats.distributions.Rand
import com.google.common.io.Files
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.DataFrame
import org.scalatest.flatspec._
import org.scalatest.matchers._

class LinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {

  val delta = 1e-5
  lazy val data: DataFrame = LinearRegressionTest._data
  lazy val vectors: Seq[Vector] = LinearRegressionTest._vectors
  lazy val result: Vector = LinearRegressionTest._result

  "Model" should "transform" in {
    val model: LinearRegressionModel = new LinearRegressionModel(
      w = Vectors.dense(0.3, 1.2, -0.5, 0.95).toDense
    ).setInputCol("features")
      .setOutputCol("features")

    validateModel(model)
  }

  "Estimator" should "fit and transform" in {
    val estimator = new LinearRegression(200, 0.1)
      .setInputCol("features")
      .setOutputCol("features")

    val model = estimator.fit(data)

    for (i <- 0 to 3) {
      model.w(i) should be(this.result(i) +- delta)
    }
    validateModel(model)
  }

  private def validateModel(model: LinearRegressionModel) = {
    for (i <- 0 to 3) {
      model.w(i) should be(this.result(i) +- delta)
    }
    val res = model.transform(data)
    val res_list = res.select("features").rdd.map(row => row(0)).collect().toList
    val data_list = data.select("features").rdd.map(row => row(0)).collect().toList

    for (i <- 0 until (res.count().toInt)) {
      val a = res_list(i).toString.replace("]", "").toDouble
      val b = data_list(i).toString.split(",").toList.last.replace("]", "").toDouble
      a should be(b +- delta)
    }
  }

  "Model" should "work after re-read" in {

    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression(200, 0.1)
        .setInputCol("features")
        .setOutputCol("features")
    ))

    val model = pipeline.fit(data)

    val tmpFolder = Files.createTempDir()

    model.write.overwrite().save(tmpFolder.getAbsolutePath)

    val reRead: PipelineModel = PipelineModel.load(tmpFolder.getAbsolutePath)

    validateModel(reRead.stages(0).asInstanceOf[LinearRegressionModel])
  }
}


object LinearRegressionTest extends WithSpark {
  lazy val _result = Vectors.dense(0.3, 1.2, -0.5, 0.95)
  var arr: Array[Vector] = Array[Vector]()
  for (i <- 1 to 1e5.toInt) {
    val x = breeze.linalg.DenseVector.rand(3, Rand.gaussian(0, 1))
    var y: Double = 0
    for (i <- 0 to 2) {
      y += _result(i) * x(i)
    }
    y += _result(3)
    arr = arr :+ Vectors.dense(x.toArray :+ y)
  }
  lazy val _vectors = arr.toSeq
  lazy val _data: DataFrame = {
    import sqlc.implicits._
    _vectors.map(x => Tuple1(x)).toDF("features")
  }
}
