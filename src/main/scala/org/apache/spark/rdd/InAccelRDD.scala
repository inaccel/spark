package org.apache.spark.rdd

import com.inaccel.coral.InAccel
import com.inaccel.coral.msg.Request
import com.inaccel.coral.shm.{SharedFloatMatrix, SharedIntMatrix}

import org.apache.spark.{Partition, SparkEnv, TaskContext}
import org.apache.spark.mllib.linalg.{Vector => MLlibVector, Vectors => MLlibVectors}
import org.apache.spark.mllib.regression.{LabeledPoint => MLlibLabeledPoint}
import org.apache.spark.util.Utils

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag
import scala.util.Random

class InAccelPartition(val label: SharedIntMatrix, val features: SharedFloatMatrix) extends Serializable {

	def free(): this.type = {
		if (label != null) {
			label.free
		}

		features.free

		this
	}

	def count(): Int = {
		features.numRows
	}

	def sample(num: Int, seed: Long = Utils.random.nextLong): ArrayBuffer[MLlibVector] = {
		val random = new Random(seed)

		var result = new ArrayBuffer[MLlibVector](num)
		for (k <- 0 until num) {
			var i = random.nextInt(features.numRows)
			val featuresSample = new Array[Double](features.numCols)
			for (j <- 0 until features.numCols) {
				featuresSample(j) = features.get(i, j).toDouble
			}
			result += MLlibVectors.dense(featuresSample)
		}
		result
	}

}

class InAccelRDD[T: ClassTag](prev: RDD[T]) extends RDD[InAccelPartition](prev) {

	override def compute(split: Partition, context: TaskContext): Iterator[InAccelPartition] = {
		inaccel(firstParent[T].compute(split, context))
	}

	override def getPartitions(): Array[Partition] = {
		firstParent[T].partitions
	}

	def inaccel(iter: Iterator[T]): Iterator[InAccelPartition] = {
		val array: Array[T] = iter.toArray
		array.apply(0) match{
			case _: InAccelPartition => fromInAccelPartition(array)
			case _: MLlibLabeledPoint => fromLabeledPoint(array)
			case _: MLlibVector => fromVector(array)
			case _: Any => Iterator()
		}
	}

	def fromInAccelPartition(array: Array[T]): Iterator[InAccelPartition] = {
		array.asInstanceOf[Array[InAccelPartition]].toIterator
	}

	def fromLabeledPoint(array: Array[T]): Iterator[InAccelPartition] = {
		val dataset = array.asInstanceOf[Array[MLlibLabeledPoint]]
		val numExamples = dataset.size
		var numFeatures = dataset.apply(0).features.size

		val labelMatrix = new SharedIntMatrix(numExamples, 1).setColAttributes(0, 8).alloc

		val featuresMatrix = new SharedFloatMatrix(numExamples, numFeatures).setRowAttributes(1, 16).setColAttributes(0, 8).alloc

		for (i <- 0 until numExamples) {
			val temp = dataset.apply(i)

			labelMatrix.put(i, 0, temp.label.toInt);

			for (j <- 0 until numFeatures) {
				featuresMatrix.put(i, j, temp.features.apply(j).toFloat)
			}
			featuresMatrix.put(i, numFeatures, 1.toFloat);
		}

		Iterator(new InAccelPartition(labelMatrix, featuresMatrix))
	}

	def fromVector(array: Array[T]): Iterator[InAccelPartition] = {
		val dataset = array.asInstanceOf[Array[MLlibVector]]
		val numExamples = dataset.size
		var numFeatures = dataset.apply(0).size

		val featuresMatrix = new SharedFloatMatrix(numExamples, numFeatures).setRowAttributes(1, 16).setColAttributes(0, 8).alloc

		for (i <- 0 until numExamples) {
			val temp = dataset.apply(i)

			for (j <- 0 until numFeatures) {
				featuresMatrix.put(i, j, temp.apply(j).toFloat)
			}
			featuresMatrix.put(i, numFeatures, 1.toFloat)
		}

		Iterator(new InAccelPartition(null, featuresMatrix))
	}

	def stash(blocking: Boolean = false): InAccelRDD[InAccelPartition] = {
		val temp = new InAccelRDD(this).persist()

		if (blocking) temp.collect

		temp
	}

	def unstash(blocking: Boolean = true): InAccelRDD[InAccelPartition] = {
		val temp = new InAccelRDD(map(partition => partition.free)).unpersist()

		if (blocking) temp.collect

		temp
	}

	override def count(): Long = {
		map(partition => partition.count).sum.toLong
	}

	def sample(num: Int): ArrayBuffer[MLlibVector] = {
		map(partition => partition.sample(num)).first
	}

}

object InAccelRDD extends Serializable {

	def gradients(partition: InAccelPartition, weights: Array[Float]): Array[Float] = {
		val numExamples = partition.features.numRowsWithAttributes
		val numFeatures = partition.features.numCols
		val numClasses = (weights.size / (numFeatures + 1)).toInt

		val result = new Array[Float](numClasses * (numFeatures + 1))

		val fpga_weights = new SharedFloatMatrix(numClasses, numFeatures + 1)
			.setRowAttributes(0, 16).alloc

		val fpga_gradients = new SharedFloatMatrix(numClasses, numFeatures + 1)
			.setRowAttributes(0, 16).alloc

		for (k <- 0 until numClasses) {
			for (j <- 0 until numFeatures + 1) {
				fpga_weights.put(k, j, weights(k * (numFeatures + 1) + j))
			}
		}

		InAccel.wait(InAccel.submit(
			new Request("com.inaccel.ml.LogisticRegression.Gradients")
				.arg(partition.label)
				.arg(partition.features)
				.arg(fpga_weights)
				.arg(fpga_gradients)
				.arg(numClasses)
				.arg(numFeatures)
				.arg(numExamples)
		))

		for (k <- 0 until numClasses) {
			for (j <- 0 until numFeatures + 1) {
				result(k * (numFeatures + 1) + j) = fpga_gradients.get(k, j)
			}
		}

		fpga_gradients.free

		fpga_weights.free

		result
	}

	def sums_counts(partition: InAccelPartition, centers: Array[MLlibVector]): Array[Float] = {
		val numExamples = partition.features.numRowsWithAttributes
		val numFeatures = partition.features.numCols
		val numClusters = centers.size.toInt

		val result = new Array[Float] (numClusters * (numFeatures + 1))

		val fpga_centers = if (numClusters <= 207) { new SharedFloatMatrix(numClusters, numFeatures).setRowAttributes(1, 16).alloc } else { new SharedFloatMatrix(numFeatures, numClusters).setRowAttributes(0, 16).setColAttributes(1,16).alloc }

		val fpga_sums_counts = new SharedFloatMatrix(numClusters, numFeatures + 1).setRowAttributes(0, 16).alloc

		for (k <- 0 until numClusters) {
			for (j <- 0 until numFeatures) {
				if (numClusters <= 207) {
					fpga_centers.put(k, j, centers(k).apply(j).toFloat)
				} else{
					fpga_centers.put(j, k, centers(k).apply(j).toFloat)
				}
			}
		}

		if (numClusters <= 207) {
			InAccel.wait(InAccel.submit(
				new Request("com.inaccel.ml.KMeans.Centroids")
					.arg(partition.features)
					.arg(fpga_centers)
					.arg(fpga_sums_counts)
					.arg(numClusters)
					.arg(numFeatures)
					.arg(numExamples)
			))
		} else {
			InAccel.wait(InAccel.submit(
				new Request("com.inaccel.ml.KMeans.Centroids1")
					.arg(partition.features)
					.arg(fpga_centers)
					.arg(fpga_sums_counts)
					.arg(numClusters)
					.arg(numFeatures)
					.arg(numExamples)
			))
		}

		for (k <- 0 until numClusters) {
			for (j <- 0 until numFeatures + 1) {
				result(k * (numFeatures + 1) + j) = fpga_sums_counts.get(k, j)
			}
		}

		fpga_sums_counts.free

		fpga_centers.free

		result
	}

}
