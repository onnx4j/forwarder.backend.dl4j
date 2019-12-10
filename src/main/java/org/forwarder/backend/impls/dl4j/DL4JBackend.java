package org.forwarder.backend.impls.dl4j;

import java.util.Arrays;

import org.forwarder.Backend;
import org.forwarder.Model;
import org.forwarder.Session;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.onnx4j.tensor.Shape;

public class DL4JBackend extends Backend<INDArray> {

	public static final String BACKEND_NAME = "DL4J";

	public DL4JBackend() {
		super();
	}

	public DL4JBackend(Model model) {
		super(model);
	}

	@Override
	public String getName() {
		return BACKEND_NAME;
	}

	@Override
	public void disposeBackendTensor(INDArray backendTensor) {
		backendTensor.close();
	}

	@Override
	public INDArray toBackendTensor(org.onnx4j.Tensor rawTensor) {
		DataBuffer dataBuffer = null;
		switch (rawTensor.getDataType()) {
		case UINT8:
		case UINT16:
		case UINT32:
		case INT8:
		case INT16:
		case INT32:
			dataBuffer = Nd4j.createBuffer(rawTensor.getData(), DataType.INT,
					(int) rawTensor.getElementSize());
			break;
		case UINT64:
		case INT64:
			dataBuffer = Nd4j.createBuffer(rawTensor.getData(), DataType.LONG,
					(int) rawTensor.getElementSize());
			break;
		case FLOAT16:
			dataBuffer = Nd4j.createBuffer(rawTensor.getData(), DataType.HALF,
					(int) rawTensor.getElementSize());
			break;
		case FLOAT:
			dataBuffer = Nd4j.createBuffer(rawTensor.getData(), DataType.FLOAT,
					(int) rawTensor.getElementSize());
			break;
		case DOUBLE:
			dataBuffer = Nd4j.createBuffer(rawTensor.getData(), DataType.DOUBLE,
					(int) rawTensor.getElementSize());
			break;
		default:
			throw new UnsupportedOperationException(
					String.format("DataType \"%s\" not be supported in ND4J", rawTensor.getDataType()));
		}

		int shape[] = Arrays.stream(rawTensor.getShape()).mapToInt(i -> (int) i).toArray();
		INDArray ndArray = Nd4j.create(dataBuffer, shape);
		return ndArray;
	}

	@Override
	public org.onnx4j.Tensor toTensor(INDArray backendTensor) {
		org.onnx4j.tensor.DataType nativeDataType = null;
		switch (backendTensor.data().dataType()) {
		case INT:
			nativeDataType = org.onnx4j.tensor.DataType.INT32;
			break;
		case LONG:
			nativeDataType = org.onnx4j.tensor.DataType.INT64;
			break;
		case HALF:
			nativeDataType = org.onnx4j.tensor.DataType.FLOAT16;
			break;
		case FLOAT:
			nativeDataType = org.onnx4j.tensor.DataType.FLOAT;
			break;
		case DOUBLE:
			nativeDataType = org.onnx4j.tensor.DataType.DOUBLE;
			break;
		default:
			throw new UnsupportedOperationException(
					String.format("DataType \"%s\" not be supported in ONNX4J", backendTensor.data().dataType()));
		}
		
		return new org.onnx4j.Tensor(
					nativeDataType, 
					Shape.create(backendTensor.shape()),
					backendTensor.data().asNio()
				);

		/*return new org.onnx4j.Tensor(nativeDataType,
				Shape.create(backendTensor.shape()),
				Memory.builder(backendTensor.data().asNio()).build());*/
	}

	@Override
	public Session<INDArray> newSession() {
		return new DL4JSession(this);
	}

}