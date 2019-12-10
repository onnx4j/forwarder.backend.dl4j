package org.forwarder.backend.impls.dl4j.opsets.v6.ops;

import org.forwarder.backend.impls.dl4j.opsets.v1.ops.DL4JMulV1;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Broadcast;
import org.onnx4j.opsets.v6.ops.MulV6;

public class DL4JMulV6 extends DL4JMulV1 implements MulV6<INDArray> {

	@Override
	public INDArray mul(INDArray a, INDArray b, Long axis, Long broadcast) {
		if (broadcast == 1L) {
			return Broadcast.mul(a, b, a, axis.intValue());
		} else {
			return a.mul(b);
		}
	}

}
