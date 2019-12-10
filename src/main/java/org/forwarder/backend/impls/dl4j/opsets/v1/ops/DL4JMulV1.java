package org.forwarder.backend.impls.dl4j.opsets.v1.ops;

import java.util.List;

import org.forwarder.backend.impls.dl4j.opsets.DL4JOperator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Broadcast;
import org.onnx4j.opsets.v1.ops.MulV1;

public class DL4JMulV1 extends DL4JOperator implements MulV1<INDArray> {

	@Override
	public INDArray mul(INDArray a, INDArray b, Long axis, Long broadcast, List<Long> consumedInputs) {
		if (broadcast == 1L) {
			return Broadcast.mul(a, b, a, axis.intValue());
		} else {
			return a.mul(b);
		}
	}

}
