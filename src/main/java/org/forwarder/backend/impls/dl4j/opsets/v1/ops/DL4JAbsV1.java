package org.forwarder.backend.impls.dl4j.opsets.v1.ops;

import java.util.Collections;

import org.forwarder.backend.impls.dl4j.DL4JSession;
import org.forwarder.backend.impls.dl4j.opsets.DL4JOperator;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.onnx4j.opsets.v1.ops.AbsV1;

public class DL4JAbsV1 extends DL4JOperator implements AbsV1<INDArray> {

	@Override
	public INDArray abs(INDArray x) {
		SameDiff sameDiff = DL4JSession.get();
		SDVariable abs = sameDiff.math.abs(sameDiff.constant(x));
		return sameDiff.outputSingle(Collections.<String, INDArray>emptyMap(), abs.getVarName());
	}

}
