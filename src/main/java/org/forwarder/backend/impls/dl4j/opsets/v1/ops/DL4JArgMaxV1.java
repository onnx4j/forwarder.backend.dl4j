package org.forwarder.backend.impls.dl4j.opsets.v1.ops;

import org.forwarder.backend.impls.dl4j.opsets.DL4JOperator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.onnx4j.opsets.v1.ops.ArgMaxV1;

public class DL4JArgMaxV1 extends DL4JOperator implements ArgMaxV1<INDArray> {

	@Override
	public INDArray argmax(INDArray x0, int axis, int keepdims) {
		return x0.argMax(axis);
	}

}
