package org.apache.spark.ml.inaccel.util

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.recommendation.ALS.Rating
import org.apache.spark.mllib.linalg.{Vector => MLlibVector, Vectors => MLlibVectors}
import org.apache.spark.mllib.regression.{LabeledPoint => MLlibLabeledPoint}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.sql.functions.col

object DatasetUtils {

	def toLabeledPointRDD(dataset: Dataset[_], labelCol: String, featuresCol: String): RDD[MLlibLabeledPoint] = {
		dataset.select(col(labelCol), col(featuresCol)).rdd.map{
			case Row(label: Double, features: Vector) => MLlibLabeledPoint(label, MLlibVectors.fromML(features))
			case Row(label: Float, features: Vector) => MLlibLabeledPoint(label.toDouble, MLlibVectors.fromML(features))
			case Row(label: Int, features: Vector) => MLlibLabeledPoint(label.toDouble, MLlibVectors.fromML(features))
			case Row(label: Long, features: Vector) => MLlibLabeledPoint(label.toDouble, MLlibVectors.fromML(features))
		}
	}

	def toRatingRDD(dataset: Dataset[_], userCol: String, itemCol: String, ratingCol: String): RDD[Rating[Int]] = {
		dataset.select(col(userCol), col(itemCol), col(ratingCol)).rdd.map{
			case Row(user: Int, item: Int, rating: Double) => Rating(user, item, rating.toFloat)
			case Row(user: Int, item: Int, rating: Float) => Rating(user, item, rating)
			case Row(user: Int, item: Int, rating: Int) => Rating(user, item, rating.toFloat)
			case Row(user: Int, item: Int, rating: Long) => Rating(user, item, rating.toFloat)
		}
	}

	def toVectorRDD(dataset: Dataset[_], featuresCol: String): RDD[MLlibVector] = {
		dataset.select(col(featuresCol)).rdd.map{
			case Row(features: Vector) => MLlibVectors.fromML(features)
		}
	}

}
