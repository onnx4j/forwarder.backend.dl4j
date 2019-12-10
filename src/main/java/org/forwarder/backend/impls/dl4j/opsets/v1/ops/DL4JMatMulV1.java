package org.forwarder.backend.impls.dl4j.opsets.v1.ops;

import org.forwarder.backend.impls.dl4j.opsets.DL4JOperator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.onnx4j.opsets.v1.ops.MatMulV1;

public class DL4JMatMulV1 extends DL4JOperator implements MatMulV1<INDArray> {

	@Override
	public INDArray matmul(INDArray x0, INDArray x1) {
		return x0.mmul(x1);
	}

}
