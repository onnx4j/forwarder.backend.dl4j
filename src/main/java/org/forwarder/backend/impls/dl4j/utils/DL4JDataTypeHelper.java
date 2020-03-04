package org.forwarder.backend.impls.dl4j.utils;

import org.nd4j.linalg.api.buffer.DataType;

import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;

public class DL4JDataTypeHelper {

	protected static BiMap<org.onnx4j.tensor.DataType, org.nd4j.linalg.api.buffer.DataType> DATATYPE_ONNX4J_DL4J_MAP;
	protected static BiMap<org.nd4j.linalg.api.buffer.DataType, org.onnx4j.tensor.DataType> DATATYPE_DL4J_ONNX4J_MAP;

	static {
		DATATYPE_ONNX4J_DL4J_MAP = HashBiMap.create();
		DATATYPE_ONNX4J_DL4J_MAP.put(org.onnx4j.tensor.DataType.UINT8, org.nd4j.linalg.api.buffer.DataType.UBYTE);
		DATATYPE_ONNX4J_DL4J_MAP.put(org.onnx4j.tensor.DataType.UINT16, org.nd4j.linalg.api.buffer.DataType.UINT16);
		DATATYPE_ONNX4J_DL4J_MAP.put(org.onnx4j.tensor.DataType.UINT32, org.nd4j.linalg.api.buffer.DataType.UINT32);
		DATATYPE_ONNX4J_DL4J_MAP.put(org.onnx4j.tensor.DataType.UINT64, org.nd4j.linalg.api.buffer.DataType.UINT64);
		DATATYPE_ONNX4J_DL4J_MAP.put(org.onnx4j.tensor.DataType.INT8, org.nd4j.linalg.api.buffer.DataType.BYTE);
		DATATYPE_ONNX4J_DL4J_MAP.put(org.onnx4j.tensor.DataType.INT16, org.nd4j.linalg.api.buffer.DataType.SHORT);
		DATATYPE_ONNX4J_DL4J_MAP.put(org.onnx4j.tensor.DataType.INT32, org.nd4j.linalg.api.buffer.DataType.INT);
		DATATYPE_ONNX4J_DL4J_MAP.put(org.onnx4j.tensor.DataType.INT64, org.nd4j.linalg.api.buffer.DataType.LONG);
		DATATYPE_ONNX4J_DL4J_MAP.put(org.onnx4j.tensor.DataType.FLOAT16, org.nd4j.linalg.api.buffer.DataType.HALF);
		DATATYPE_ONNX4J_DL4J_MAP.put(org.onnx4j.tensor.DataType.FLOAT, org.nd4j.linalg.api.buffer.DataType.FLOAT);
		DATATYPE_ONNX4J_DL4J_MAP.put(org.onnx4j.tensor.DataType.DOUBLE, org.nd4j.linalg.api.buffer.DataType.DOUBLE);
		DATATYPE_ONNX4J_DL4J_MAP.put(org.onnx4j.tensor.DataType.BOOL, org.nd4j.linalg.api.buffer.DataType.BOOL);
		DATATYPE_ONNX4J_DL4J_MAP.put(org.onnx4j.tensor.DataType.STRING, org.nd4j.linalg.api.buffer.DataType.UTF8);

		DATATYPE_DL4J_ONNX4J_MAP = DATATYPE_ONNX4J_DL4J_MAP.inverse();
	}

	public static org.nd4j.linalg.api.buffer.DataType toDl4jDataType(org.onnx4j.tensor.DataType onnx4jDataType) {
		DataType datatype = DATATYPE_ONNX4J_DL4J_MAP.get(onnx4jDataType);
		if (datatype == null)
			throw new UnsupportedOperationException(
					String.format("DataType \"%s\" not be supported in ND4J", onnx4jDataType));
		else
			return datatype;
	}

	public static org.onnx4j.tensor.DataType toOnnx4jDataType(org.nd4j.linalg.api.buffer.DataType dl4jDataType) {
		org.onnx4j.tensor.DataType datatype = DATATYPE_DL4J_ONNX4J_MAP.get(dl4jDataType);
		if (datatype == null)
			throw new UnsupportedOperationException(
					String.format("Nd4j data type \"%s\" not be supported in Onnx4j", dl4jDataType));
		else
			return datatype;
	}

	public static void ensureNotContainUnexceptedType(DataType currentType, DataType[] dl4jUnexceptedTypes) {
		for (DataType dl4jUnexceptedType : dl4jUnexceptedTypes) {
			if (dl4jUnexceptedType.equals(currentType))
				throw new RuntimeException(String.format("Nd4j data type \"\" is invalid type in this operation"));
		}
	}

}
