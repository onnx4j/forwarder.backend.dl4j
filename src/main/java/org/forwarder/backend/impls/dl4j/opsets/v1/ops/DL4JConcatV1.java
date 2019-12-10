package org.forwarder.backend.impls.dl4j.opsets.v1.ops;

import java.util.Collections;
import java.util.List;

import org.forwarder.backend.impls.dl4j.DL4JSession;
import org.forwarder.backend.impls.dl4j.opsets.DL4JOperator;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.onnx4j.opsets.v1.ops.ConcatV1;

public class DL4JConcatV1 extends DL4JOperator implements ConcatV1<INDArray> {

	@Override
	public INDArray concat(List<INDArray> inputs, Long axis) {
		SameDiff sameDiff = DL4JSession.get();

		SDVariable[] sdInputs = new SDVariable[inputs.size()];
		for (int n = 0; n < sdInputs.length; n++) {
			sdInputs[n] = sameDiff.constant(inputs.get(n));
		}

		SDVariable concat = sameDiff.concat(axis.intValue(), sdInputs);
		return sameDiff.outputSingle(Collections.<String, INDArray>emptyMap(), concat.getVarName());
	}

}
