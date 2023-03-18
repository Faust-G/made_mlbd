package org.apache.spark.ml.made

import com.google.common.io.Files
import org.scalatest._
import flatspec._
import matchers._
import org.apache.spark.ml
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.linalg.{DenseVector, Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.{DataFrame, Dataset, Encoder, SparkSession}
import breeze.stats.distributions.Rand
import org.apache.spark.ml.made.StandardScalerTest._result
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder

class StandardScalerTest extends AnyFlatSpec with should.Matchers with WithSpark {

  val delta = 1e-5
  lazy val data: DataFrame = StandardScalerTest._data
  lazy val vectors: Seq[Vector] = StandardScalerTest._vectors
  lazy val result: Vector = StandardScalerTest._result

  "Model" should "scale input data" in {
    val model: StandardScalerModel = new StandardScalerModel(
      w = Vectors.dense(0.3, 1.2, -0.5, 0.95).toDense
    ).setInputCol("features")
      .setOutputCol("features")

    validateModel(model)
  }

  "Estimator" should "fit and transform" in {
    val estimator = new StandardScaler(200, 0.1)
      .setInputCol("features")
      .setOutputCol("features")

    val model = estimator.fit(data)

    for (i <- 0 to 3) {
      model.w(i) should be (this.result(i) +- delta)
    }
    validateModel(model)
  }
//
//  "Estimator" should "transform" in {
//    val estimator = new StandardScaler(200, 0.1)
//      .setInputCol("features")
//      .setOutputCol("features")
////    data.show()
//    //    print("dfsfsfs",data.count(), len(data.columns))
//    val model = estimator.fit(data)
//    //    print(model.w)
//    for (i <- 1 to 3) {
//      model.w(i) should be(this.result(i) +- delta)
//    }
//  }
//
//  "Estimator" should "should produce functional model" in {
//    val estimator = new StandardScaler()
//      .setInputCol("features")
//      .setOutputCol("features")
//
//    val model = estimator.fit(data)
//
//    validateModel(model, model.transform(data))
//  }
//
private def validateModel(model: StandardScalerModel) = {
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
      new StandardScaler(200,0.1)
        .setInputCol("features")
        .setOutputCol("features")
    ))

    val model = pipeline.fit(data)

    val tmpFolder = Files.createTempDir()

    model.write.overwrite().save(tmpFolder.getAbsolutePath)

    val reRead: PipelineModel = PipelineModel.load(tmpFolder.getAbsolutePath)

    validateModel(reRead.stages(0).asInstanceOf[StandardScalerModel])
  }

  "Al" should "work after re-read" in {


    val pipeline = new Pipeline().setStages(Array(
      new StandardScaler(200,0.1)
        .setInputCol("features")
        .setOutputCol("features")
    ))

    //    val model = pipeline.fit(data)

    val tmpFolder = Files.createTempDir()

    pipeline.write.overwrite().save(tmpFolder.getAbsolutePath)
    val reRead = Pipeline.load(tmpFolder.getAbsolutePath)
//
//    val model = reRead.fit(data).stages(0).asInstanceOf[StandardScalerModel]
  }
}


object StandardScalerTest extends WithSpark {
  lazy val _result = Vectors.dense(0.3, 1.2, -0.5, 0.95)
  var arr: Array[Vector] = Array[Vector]()
  for (i <- 1 to 1e5.toInt) {
    val x = breeze.linalg.DenseVector.rand(3, Rand.gaussian(0, 1))
    var y:Double = 0
    for (i <- 0 to 2){
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
