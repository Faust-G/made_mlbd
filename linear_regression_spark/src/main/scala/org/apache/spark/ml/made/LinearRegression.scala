package org.apache.spark.ml.made

import breeze.linalg._
import breeze.stats.distributions.Rand
import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.ml.linalg.{DenseVector, Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Encoder}

import scala.util.control.Breaks._

trait LinearRegressionParams extends HasInputCol with HasOutputCol {
  def setInputCol(value: String): this.type = set(inputCol, value)

  def setOutputCol(value: String): this.type = set(outputCol, value)

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getInputCol, new VectorUDT())

    if (schema.fieldNames.contains($(outputCol))) {
      SchemaUtils.checkColumnType(schema, getOutputCol, new VectorUDT())
      schema
    } else {
      SchemaUtils.appendColumn(schema, schema(getInputCol).copy(name = getOutputCol))
    }
  }
}

class LinearRegression(override val uid: String, val max_iter: Int, var alpha: Double) extends Estimator[LinearRegressionModel] with LinearRegressionParams
  with DefaultParamsWritable {

  def this(max_iter: Int, alpha: Double) = this(Identifiable.randomUID("standardScaler"), max_iter, alpha)

  def this() = this(Identifiable.randomUID("standardScaler"), 200, 0.01)

  override def fit(dataset: Dataset[_]): LinearRegressionModel = {


    implicit val encoder: Encoder[Vector] = ExpressionEncoder()

    val vectors: Dataset[Vector] = dataset.select(dataset($(inputCol)).as[Vector])
    val dim: Int = AttributeGroup.fromStructField((dataset.schema($(inputCol)))).numAttributes.getOrElse(
      vectors.first().size
    )
    val points = vectors.rdd.cache()
    val wp: breeze.linalg.DenseVector[Double] = breeze.linalg.DenseVector.rand(dim, Rand.gaussian(0, 1))
    val sz: Double = dataset.count()
    breakable {
      for (i <- 1 to max_iter) {
        val gradient = points.map(p => {
          var arr: Array[Double] = p.toArray
          val y = arr.last
          arr = arr.dropRight(1)
          arr :+= 1.0
          val x: breeze.linalg.DenseVector[Double] = breeze.linalg.DenseVector(arr)
          (2.0 * ((wp.dot(x)) - y) * x)
        }).reduce(_ + _).toDenseVector
        if (norm(gradient) < 1e-4) break
        wp -= alpha / sz * gradient
      }
    }

    val vectout: DenseVector = Vectors.dense(wp.toArray).toDense
    copyValues(new LinearRegressionModel(vectout)).setParent(this)
  }

  override def copy(extra: ParamMap): Estimator[LinearRegressionModel] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

}

object LinearRegression extends DefaultParamsReadable[LinearRegression]

class LinearRegressionModel private[made](
                                           override val uid: String,
                                           val w: DenseVector)
  extends Model[LinearRegressionModel] with LinearRegressionParams with MLWritable {


  private[made] def this(w: Vector) =
    this(Identifiable.randomUID("standardScalerModel"), w.toDense)

  override def copy(extra: ParamMap): LinearRegressionModel = copyValues(
    new LinearRegressionModel(w), extra)

  override def transform(dataset: Dataset[_]): DataFrame = {
    val bw = w.asBreeze
    val transformUdf =
      dataset.sqlContext.udf.register(uid + "_transform",
        (x: Vector) => {
          val idx = bw.length - 1
          x.asBreeze(0 until idx).dot(bw(0 until idx)) + bw(idx)
        })

    dataset.withColumn($(outputCol), transformUdf(dataset($(inputCol))))
  }

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  override def write: MLWriter = new DefaultParamsWriter(this) {
    override protected def saveImpl(path: String): Unit = {

      super.saveImpl(path)

      val vectors = Tuple1(w.asInstanceOf[Vector])
      sqlContext.createDataFrame(Seq(vectors)).write.parquet(path + "/vectors")
    }
  }
}

object LinearRegressionModel extends MLReadable[LinearRegressionModel] {
  override def read: MLReader[LinearRegressionModel] = new MLReader[LinearRegressionModel] {
    override def load(path: String): LinearRegressionModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc)

      val vectors = sqlContext.read.parquet(path + "/vectors")


      implicit val encoder: Encoder[Vector] = ExpressionEncoder()

      val w = vectors.select(vectors("_1").as[Vector]).first()

      val model = new LinearRegressionModel(w)
      metadata.getAndSetParams(model)
      model
    }
  }
}
