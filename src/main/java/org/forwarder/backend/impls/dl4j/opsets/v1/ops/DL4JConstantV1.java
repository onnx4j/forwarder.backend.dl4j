package org.forwarder.backend.impls.dl4j.opsets.v1.ops;

import org.forwarder.backend.impls.dl4j.DL4JSession;
import org.forwarder.backend.impls.dl4j.opsets.DL4JOperator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.onnx4j.Tensor;
import org.onnx4j.opsets.v1.ops.ConstantV1;

public class DL4JConstantV1 extends DL4JOperator implements ConstantV1<INDArray> {

	@Override
	public INDArray constant(Tensor x0) {
		return DL4JSession.getSession().getBackend().toBackendTensor(x0);
	}

}
