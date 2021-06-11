package org.apache.spark.rdd

import com.inaccel.coral.InAccelByteBufAllocator
import io.netty.buffer.ByteBuf
import java.io.{IOException, ObjectInputStream, ObjectOutputStream, ObjectStreamException}

@SerialVersionUID(564731749280392836L)
class MatrixBuf(val rows: Int, val cols:Int, val typeSize: Int) extends Serializable {
	@transient
	var buf: ByteBuf = null

	/** The number of padding rows. */
	private var rowsPadding: Int = 0

	/** The number of meta rows. */
	private var rowsMeta: Int = 0

	/** The number of padding columns. */
	private var colsPadding: Int = 0

	/** The number of meta columns. */
	private var colsMeta: Int = 0

	/**
	* Allocates an InAccel Netty ByteBuf according to cols' and rows' options provided
	* @return This matrix.
	*/
	def alloc(): MatrixBuf = {
		buf = InAccelByteBufAllocator.DEFAULT.buffer(numColsWithAttributes * numRowsWithAttributes * typeSize)

		this
	}

	/**
	* Sets the attributes of the row.
	* @param colsMeta The number of meta columns.
	* @param vector The vectorization factor.
	* @return This matrix.
	*/
	def setRowAttributes(colsMeta: Int, vector: Int): MatrixBuf = {
		this.colsMeta = colsMeta

		colsPadding = (vector - ((cols + colsMeta) % vector)) % vector

		this
	}

	/**
	* Sets the attributes of the column.
	* @param rowsMeta The number of meta rows.
	* @param vector The vectorization factor.
	* @return This matrix.
	*/
	def setColAttributes(rowsMeta: Int, vector: Int): MatrixBuf = {
		this.rowsMeta = rowsMeta

		rowsPadding = (vector - ((rows + rowsMeta) % vector)) % vector

		this
	}

	/**
	* Returns row dimension.
	* @return The number of rows.
	*/
	def numRows(): Int = {
		rows
	}

	/**
	* Returns column dimension.
	* @return The number of columns.
	*/
	def numCols(): Int = {
		cols
	}

	/**
	* Returns meta row dimension.
	* @return The number of meta rows.
	*/
	def numMetaRows(): Int = {
		rowsMeta
	}

	/**
	* Returns meta column dimension.
	* @return The number of meta columns.
	*/
	def numMetaCols(): Int = {
		colsMeta
	}

	/**
	* Returns padding row dimension.
	* @return The number of padding rows.
	*/
	def numPaddingRows(): Int = {
		rowsPadding
	}

	/**
	* Returns padding column dimension.
	* @return The number of padding columns.
	*/
	def numPaddingCols(): Int = {
		colsPadding
	}

	/**
	* Returns complete row dimension.
	* @return The number of rows, including meta and padding.
	*/
	def numRowsWithAttributes(): Int = {
		rows + rowsMeta + rowsPadding
	}

	/**
	* Returns complete column dimension.
	* @return The number of cols, including meta and padding.
	*/
	def numColsWithAttributes(): Int = {
		cols + colsMeta + colsPadding
	}

	def getBoolean(i: Int, j: Int): Boolean = {
		buf.getBoolean(typeSize * (i * numColsWithAttributes + j))
	}

	def getByte(i: Int, j: Int): Byte = {
		buf.getByte(typeSize * (i * numColsWithAttributes + j))
	}

	def getDouble(i: Int, j: Int): Double = {
		buf.getDouble(typeSize * (i * numColsWithAttributes + j))
	}

	def getFloat(i: Int, j: Int): Float = {
		buf.getFloat(typeSize * (i * numColsWithAttributes + j))
	}

	def getInt(i: Int, j: Int): Int = {
		buf.getInt(typeSize * (i * numColsWithAttributes + j))
	}

	def getLong(i: Int, j: Int): Long = {
		buf.getLong(typeSize * (i * numColsWithAttributes + j))
	}

	def getMedium(i: Int, j: Int): Int = {
		buf.getMedium(typeSize * (i * numColsWithAttributes + j))
	}

	def getShort(i: Int, j: Int): Short = {
		buf.getShort(typeSize * (i * numColsWithAttributes + j))
	}

	def setBoolean(i: Int, j: Int, value: Boolean): Unit = {
		buf.setBoolean(typeSize * (i * numColsWithAttributes + j), Boolean.box(value))
	}

	def setByte(i: Int, j: Int, value: Byte): Unit = {
		buf.setByte(typeSize * (i * numColsWithAttributes + j), Int.box(value.toInt))
	}

	def setChar(i: Int, j: Int, value: Char): Unit = {
		buf.setChar(typeSize * (i * numColsWithAttributes + j), Int.box(value.toInt))
	}

	def setDouble(i: Int, j: Int, value: Double): Unit = {
		buf.setDouble(typeSize * (i * numColsWithAttributes + j), Double.box(value))
	}

	def setFloat(i: Int, j: Int, value: Float): Unit = {
		buf.setFloat(typeSize * (i * numColsWithAttributes + j), Float.box(value))
	}

	def setInt(i: Int, j: Int, value: Int): Unit = {
		buf.setInt(typeSize * (i * numColsWithAttributes + j), Int.box(value))
	}

	def setLong(i: Int, j: Int, value: Long): Unit = {
		buf.setLong(typeSize * (i * numColsWithAttributes + j), Long.box(value))
	}

	def setMedium(i: Int, j: Int, value: Int): Unit = {
		buf.setMedium(typeSize * (i * numColsWithAttributes + j), Int.box(value))
	}

	def setShort(i: Int, j: Int, value: Short): Unit = {
		buf.setShort(typeSize * (i * numColsWithAttributes + j), Int.box(value.toInt))
	}

	@throws(classOf[IOException])
	private def writeObject(out: ObjectOutputStream): Unit = {
		out.defaultWriteObject

		buf.writerIndex(buf.capacity)
		buf.readBytes(out, buf.capacity)
	}

	@throws(classOf[IOException])
	@throws(classOf[ClassNotFoundException])
	private def readObject(in: ObjectInputStream): Unit = {
		in.defaultReadObject

		alloc
		buf.writeBytes(in, buf.capacity)
	}

}
