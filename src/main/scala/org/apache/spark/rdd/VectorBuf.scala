package org.apache.spark.rdd

import com.inaccel.coral.InAccelByteBufAllocator
import io.netty.buffer.ByteBuf
import java.io.{IOException, ObjectInputStream, ObjectOutputStream, ObjectStreamException}

@SerialVersionUID(564731749280392837L)
class VectorBuf(val cols:Int, val typeSize: Int) extends Serializable {
	@transient
	var buf: ByteBuf = null

	/** The number of padding columns. */
	private var colsPadding: Int = 0

	/** The number of meta columns. */
	private var colsMeta: Int = 0

	/**
	* Allocates an InAccel Netty ByteBuf according to cols' and rows' options provided
	* @return This vector.
	*/
	def alloc(): VectorBuf = {
		buf = InAccelByteBufAllocator.DEFAULT.buffer(numColsWithAttributes * typeSize)

		this
	}

	/**
	* Sets the attributes of the row.
	* @param colsMeta The number of meta columns.
	* @param vector The vectorization factor.
	* @return This vector.
	*/
	def setRowAttributes(colsMeta: Int, vector: Int): VectorBuf = {
		this.colsMeta = colsMeta

		colsPadding = (vector - ((cols + colsMeta) % vector)) % vector

		this
	}

	/**
	* Returns column dimension.
	* @return The number of columns.
	*/
	def numCols(): Int = {
		cols
	}

	/**
	* Returns meta column dimension.
	* @return The number of meta columns.
	*/
	def numMetaCols(): Int = {
		colsMeta
	}

	/**
	* Returns padding column dimension.
	* @return The number of padding columns.
	*/
	def numPaddingCols(): Int = {
		colsPadding
	}

	/**
	* Returns complete column dimension.
	* @return The number of cols, including meta and padding.
	*/
	def numColsWithAttributes(): Int = {
		cols + colsMeta + colsPadding
	}

	def getBoolean(i: Int): Boolean = {
		buf.getBoolean(typeSize * i)
	}

	def getByte(i: Int): Byte = {
		buf.getByte(typeSize * i)
	}

	def getDouble(i: Int): Double = {
		buf.getDouble(typeSize * i)
	}

	def getFloat(i: Int): Float = {
		buf.getFloat(typeSize * i)
	}

	def getInt(i: Int): Int = {
		buf.getInt(typeSize * i)
	}

	def getLong(i: Int): Long = {
		buf.getLong(typeSize * i)
	}

	def getMedium(i: Int): Int = {
		buf.getMedium(typeSize * i)
	}

	def getShort(i: Int): Short = {
		buf.getShort(typeSize * i)
	}

	def setBoolean(i: Int, value: Boolean): Unit = {
		buf.setBoolean(typeSize * i, Boolean.box(value))
	}

	def setByte(i: Int, value: Byte): Unit = {
		buf.setByte(typeSize * i, Int.box(value.toInt))
	}

	def setChar(i: Int, value: Char): Unit = {
		buf.setChar(typeSize * i, Int.box(value.toInt))
	}

	def setDouble(i: Int, value: Double): Unit = {
		buf.setDouble(typeSize * i, Double.box(value))
	}

	def setFloat(i: Int, value: Float): Unit = {
		buf.setFloat(typeSize * i, Float.box(value))
	}

	def setInt(i: Int, value: Int): Unit = {
		buf.setInt(typeSize * i, Int.box(value))
	}

	def setLong(i: Int, value: Long): Unit = {
		buf.setLong(typeSize * i, Long.box(value))
	}

	def setMedium(i: Int, value: Int): Unit = {
		buf.setMedium(typeSize * i, Int.box(value))
	}

	def setShort(i: Int, value: Short): Unit = {
		buf.setShort(typeSize * i, Int.box(value.toInt))
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
