package org.apache.spark.ml.inaccel.classification

import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.mllib.linalg.{DenseMatrix => MLlibDenseMatrix, DenseVector => MLlibDenseVector}
import org.apache.spark.mllib.regression.{LabeledPoint => MLlibLabeledPoint}
import org.apache.spark.rdd.{InAccelRDD, RDD}
import org.apache.spark.ui.inaccel.{ConsoleProgressBar, ConsoleTimeReporter}

class LogisticRegression extends Serializable {

	var numClasses: Int = 0

	val numClassesMax: Int = 64

	var numFeatures: Int = 0

	val numFeaturesMax: Int = 4095

	var alpha: Double = 0.3

	var gamma: Double = 0.9

	var maxIter: Int = 100

	var tol: Double = 1E-6

	def setNumClasses(value: Int): this.type = {
		require(value <= numClassesMax, "! Unsupported number of Classes " + value + ". Falling back.")

		numClasses = value
		this
	}

	def setNumFeatures(value: Int): this.type = {
		require(value <= numFeaturesMax, "! Unsupported number of Features " + value + ". Falling back.")

		numFeatures = value
		this
	}

	def setAlpha(value: Double): this.type = {
		alpha = value
		this
	}

	def setGamma(value: Double): this.type = {
		gamma = value
		this
	}

	def setMaxIter(value: Int): this.type = {
		maxIter = value
		this
	}

	def setTol(value: Double): this.type = {
		tol = value
		this
	}

	def run(data: RDD[MLlibLabeledPoint]): LogisticRegressionModel = {
		ConsoleTimeReporter.timestamp("A")

		println("\t # numClasses:  " + numClasses)
		println("\t # numFeatures: " + numFeatures)

		val fpga_data = new InAccelRDD(data).stash()
		val numExamples = fpga_data.count

		println("\t # numExamples: " + numExamples)

		ConsoleTimeReporter.timestamp("B")

		val velocity: Array[Float] = Array.fill[Float](numClasses * (numFeatures + 1))(0)
		val weights: Array[Float] = Array.fill[Float](numClasses * (numFeatures + 1))(0)

		for (i <- 0 until maxIter) {
			ConsoleProgressBar.show(i, maxIter)

			val gradients = fpga_data.map(data => InAccelRDD.gradients(data, weights)).reduce((a, b) => a.zip(b).map{case (x, y) => x + y})

			for (k <- 0 until numClasses) {
				for (j <- 0 until (numFeatures + 1)) {
					velocity(k * (numFeatures + 1) + j) = gamma.toFloat * velocity(k * (numFeatures + 1) + j) + (alpha.toFloat / numExamples) * gradients(k * (numFeatures + 1) + j)

					weights(k * (numFeatures + 1) + j) -= velocity(k * (numFeatures + 1) + j)
				}
			}

			ConsoleProgressBar.clear
		}

		ConsoleTimeReporter.timestamp("C")

		fpga_data.unstash()

		ConsoleTimeReporter.timestamp("D")

		println("! Dataset transformation duration: " + (ConsoleTimeReporter.time("A", "B") + ConsoleTimeReporter.time("C", "D")) + " sec")
		println("! Model estimation duration: " + ConsoleTimeReporter.time("B", "C") + " sec")

		sparkModel(weights)
	}

	def sparkModel(weights: Array[Float]): LogisticRegressionModel = {
		var coefficient = new Array[Double](numClasses * numFeatures)
		var intercept = new Array[Double](numClasses)

		for (k <- 0 until numClasses) {
			for (j <- 0 until numFeatures) {
				coefficient(k * numFeatures + j) = weights(k * (numFeatures + 1) + j).toDouble
			}
			intercept(k) = weights(k * (numFeatures + 1) + numFeatures).toDouble
		}

		var coefficientMatrix = new MLlibDenseMatrix(numClasses, numFeatures, coefficient, true).asML
		var interceptVector = new MLlibDenseVector(intercept).asML

		new LogisticRegressionModel("inaccel", coefficientMatrix, interceptVector, numClasses, true)
	}

}
