package org.forwarder.backend.impls.dl4j.opsets.v1.ops;

import java.util.Collections;
import java.util.List;
import java.util.Optional;

import org.forwarder.backend.impls.dl4j.DL4JSession;
import org.forwarder.backend.impls.dl4j.opsets.DL4JOperator;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.onnx4j.opsets.v1.ops.DropoutV1;

public class DL4JDropoutV1 extends DL4JOperator implements DropoutV1<INDArray> {

	@Override
	public List<INDArray> dropout(INDArray data, Boolean isTest, Float ratio, List<Long> consumedInputs) {
		//
		// if isTest is true, run dropout in test mode where the output is
		// simply Y = X
		//
		if (isTest)
			return this.wrapMultiOutputs(Optional.of(data), Optional.empty());
		else {
			SameDiff sameDiff = DL4JSession.get();
			SDVariable sdDropout = sameDiff.nn.dropout(sameDiff.constant(data), ratio);
			return this.wrapMultiOutputs(
					Optional.of(
							sameDiff.outputSingle(Collections.<String, INDArray>emptyMap(), sdDropout.getVarName())),
					Optional.empty());
		}
	}

}
