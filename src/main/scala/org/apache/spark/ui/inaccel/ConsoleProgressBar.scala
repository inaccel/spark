package org.apache.spark.ui.inaccel

object ConsoleProgressBar {

	private val CR = '\r'

	private val TerminalWidth = {
		val COLUMNS = sys.env.getOrElse("COLUMNS", "")

		if (!COLUMNS.isEmpty) COLUMNS.toInt
		else 80
	}

	def show(value: Int, maxValue: Int): Unit = {
		val progress = (value + 1).toFloat / maxValue.toFloat
		val percent = (progress * 100).toInt;

		val header = s" ${percent}% ["
		val tailer = s"] ${(value + 1)}/${maxValue}"

		val width = TerminalWidth - header.length - tailer.length
		val head = (width * progress).toInt
		val bar = (0 until width).map{i => if (i < head) "=" else if (i == head) ">" else " "}.mkString("")

		System.err.print(CR + header + bar + tailer)
	}

	def clear(): Unit = {
		System.err.print(CR + " " * TerminalWidth + CR)
	}

}
