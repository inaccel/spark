package org.apache.spark.ui.inaccel

import scala.collection.mutable.HashMap

class ConsoleTimeReporter private() {

	val report: HashMap[String, Long] = HashMap.empty[String, Long]

	def put(identifier: String, timestamp: Long): Unit = {
		report += (identifier -> timestamp)
	}

	def get(identifier: String): Long = {
		report(identifier)
	}

}

object ConsoleTimeReporter {

	private val instance = new ConsoleTimeReporter()

	def timestamp(identifier: String): Unit = {
		instance.put(identifier, System.nanoTime)
	}

	def time(identifier0: String, identifier1: String): Double = {
		(instance.get(identifier0) - instance.get(identifier1)).abs / 1E9
	}

}
