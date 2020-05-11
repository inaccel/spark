package org.apache.spark.ml.inaccel.util

import org.apache.spark.sql.Dataset
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.functions.col
import org.apache.spark.ml.attribute._

object MetadataUtils {

	def getNumClasses(dataset: Dataset[_], labelCol: String): Int = {
		// requires StringIndexer, TODO general implementation
		try {
			dataset.schema(labelCol).metadata.getMetadata("ml_attr").getStringArray("vals").length
		} catch {
			case e: Exception => {
				Attribute.fromStructField(dataset.schema(labelCol)) match {
					case binAttr: BinaryAttribute => 2
					case nomAttr: NominalAttribute => nomAttr.getNumValues.get
					case _: NumericAttribute | UnresolvedAttribute => 0
				}
			}
		}
	}

	def getNumFeatures(dataset: Dataset[_], featuresCol: String): Int = {
		try {
			dataset.schema(featuresCol).metadata.getLong("numFeatures").toInt
		} catch {
			case e: Exception => {
				val row = dataset.select(featuresCol).head
				row(0).asInstanceOf[Vector].size
			}
		}
	}

}
