package org.apache.spark.rdd

import com.inaccel.coral.{InAccel, InAccelByteBufAllocator}
import io.netty.buffer.ByteBuf
import java.lang.Float.{BYTES => FloatBytes}

import org.apache.spark.{Partition, SparkEnv, TaskContext}
import org.apache.spark.mllib.linalg.{Vector => MLlibVector, Vectors => MLlibVectors}
import org.apache.spark.mllib.regression.{LabeledPoint => MLlibLabeledPoint}
import org.apache.spark.util.Utils

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag
import scala.util.Random

class InAccelPartition(val label: MatrixBuf, val features: MatrixBuf) extends Serializable {

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
				featuresSample(j) = features.getFloat(i, j).toDouble
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

		val labelMatrix = new MatrixBuf(numExamples, 1, Integer.BYTES).setColAttributes(0, 8).alloc

		val featuresMatrix = new MatrixBuf(numExamples, numFeatures, FloatBytes).setRowAttributes(1, 16).setColAttributes(0, 8).alloc

		for (i <- 0 until numExamples) {
			val temp = dataset.apply(i)

			labelMatrix.setInt(i, 0, temp.label.toInt);

			for (j <- 0 until numFeatures) {
				featuresMatrix.setFloat(i, j, temp.features.apply(j).toFloat)
			}
			featuresMatrix.setFloat(i, numFeatures, 1.toFloat);
		}

		Iterator(new InAccelPartition(labelMatrix, featuresMatrix))
	}

	def fromVector(array: Array[T]): Iterator[InAccelPartition] = {
		val dataset = array.asInstanceOf[Array[MLlibVector]]
		val numExamples = dataset.size
		var numFeatures = dataset.apply(0).size

		val featuresMatrix = new MatrixBuf(numExamples, numFeatures, FloatBytes).setRowAttributes(1, 16).setColAttributes(0, 8).alloc

		for (i <- 0 until numExamples) {
			val temp = dataset.apply(i)

			for (j <- 0 until numFeatures) {
				featuresMatrix.setFloat(i, j, temp.apply(j).toFloat)
			}
			featuresMatrix.setFloat(i, numFeatures, 1.toFloat)
		}

		Iterator(new InAccelPartition(null, featuresMatrix))
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

		val fpga_weights = new MatrixBuf(numClasses, numFeatures + 1, FloatBytes)
			.setRowAttributes(0, 16).alloc

		val fpga_gradients = new MatrixBuf(numClasses, numFeatures + 1, FloatBytes)
			.setRowAttributes(0, 16).alloc

		for (k <- 0 until numClasses) {
			for (j <- 0 until numFeatures + 1) {
				fpga_weights.setFloat(k, j, weights(k * (numFeatures + 1) + j))
			}
		}

		InAccel.submit(
			new InAccel.Request("com.inaccel.ml.LogisticRegression.Gradients")
				.arg(partition.label.buf)
				.arg(partition.features.buf)
				.arg(fpga_weights.buf)
				.arg(fpga_gradients.buf)
				.arg(Int.box(numClasses))
				.arg(Int.box(numFeatures))
				.arg(Int.box(numExamples))
		).get

		for (k <- 0 until numClasses) {
			for (j <- 0 until numFeatures + 1) {
				result(k * (numFeatures + 1) + j) = fpga_gradients.getFloat(k, j)
			}
		}

		fpga_gradients.buf.release

		fpga_weights.buf.release

		result
	}

	def sums_counts(partition: InAccelPartition, centers: Array[MLlibVector]): Array[Float] = {
		val numExamples = partition.features.numRowsWithAttributes
		val numFeatures = partition.features.numCols
		val numClusters = centers.size.toInt

		val result = new Array[Float] (numClusters * (numFeatures + 1))

		val fpga_centers = if (numClusters <= 207) { new MatrixBuf(numClusters, numFeatures, FloatBytes).setRowAttributes(1, 16).alloc } else { new MatrixBuf(numFeatures, numClusters, FloatBytes).setRowAttributes(0, 16).setColAttributes(1,16).alloc }

		val fpga_sums_counts = new MatrixBuf(numClusters, numFeatures + 1, FloatBytes).setRowAttributes(0, 16).alloc

		for (k <- 0 until numClusters) {
			for (j <- 0 until numFeatures) {
				if (numClusters <= 207) {
					fpga_centers.setFloat(k, j, centers(k).apply(j).toFloat)
				} else{
					fpga_centers.setFloat(j, k, centers(k).apply(j).toFloat)
				}
			}
		}

		if (numClusters <= 207) {
			InAccel.submit(
				new InAccel.Request("com.inaccel.ml.KMeans.Centroids")
					.arg(partition.features.buf)
					.arg(fpga_centers.buf)
					.arg(fpga_sums_counts.buf)
					.arg(Int.box(numClusters))
					.arg(Int.box(numFeatures))
					.arg(Int.box(numExamples))
			).get
		} else {
			InAccel.submit(
				new InAccel.Request("com.inaccel.ml.KMeans.Centroids1")
					.arg(partition.features.buf)
					.arg(fpga_centers.buf)
					.arg(fpga_sums_counts.buf)
					.arg(Int.box(numClusters))
					.arg(Int.box(numFeatures))
					.arg(Int.box(numExamples))
			).get
		}

		for (k <- 0 until numClusters) {
			for (j <- 0 until numFeatures + 1) {
				result(k * (numFeatures + 1) + j) = fpga_sums_counts.getFloat(k, j)
			}
		}

		fpga_sums_counts.buf.release

		fpga_centers.buf.release

		result
	}

}
