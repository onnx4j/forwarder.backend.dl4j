package org.forwarder.backend.impls.dl4j.opsets.v1.ops;

import java.util.Collections;
import java.util.List;

import org.forwarder.backend.impls.dl4j.DL4JSession;
import org.forwarder.backend.impls.dl4j.opsets.DL4JOperator;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.onnx4j.opsets.v1.ops.LeakyReluV1;

public class DL4JLeakyReluV1 extends DL4JOperator implements LeakyReluV1<INDArray> {

	@Override
	public INDArray leakyRelu(INDArray x, Float alpha, List<Long> consumedInputs) {
		SameDiff sameDiff = DL4JSession.get();
		SDVariable relu = sameDiff.nn.leakyRelu(sameDiff.constant(x), alpha);
		return sameDiff.outputSingle(
				Collections.<String, INDArray>emptyMap(), 
				relu.getVarName());
	}

}
