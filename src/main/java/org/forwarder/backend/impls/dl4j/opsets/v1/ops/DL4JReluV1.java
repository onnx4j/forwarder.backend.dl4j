package org.forwarder.backend.impls.dl4j.opsets.v1.ops;

import java.util.Collections;
import java.util.List;

import org.forwarder.backend.impls.dl4j.DL4JSession;
import org.forwarder.backend.impls.dl4j.opsets.DL4JOperator;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.onnx4j.opsets.v1.ops.ReluV1;

public class DL4JReluV1 extends DL4JOperator implements ReluV1<INDArray> {

	@Override
	public INDArray relu(INDArray x, List<Long> consumed_inputs) {
		SameDiff sameDiff = DL4JSession.get();
		SDVariable relu = sameDiff.nn.relu(sameDiff.constant(x), 0.0d);
		return sameDiff.outputSingle(
				Collections.<String, INDArray>emptyMap(), 
				relu.getVarName());
	}

}
