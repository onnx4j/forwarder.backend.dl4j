package org.forwarder.backend.impls.dl4j.opsets.v1.ops;

import java.util.Collections;
import java.util.List;

import org.forwarder.backend.impls.dl4j.DL4JSession;
import org.forwarder.backend.impls.dl4j.opsets.DL4JOperator;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.onnx4j.opsets.v1.ops.BatchNormalizationV1;

public class DL4JBatchNormalizationV1 extends DL4JOperator implements BatchNormalizationV1<INDArray> {

	@Override
	public INDArray[] batchNormalization(INDArray x, INDArray scale, INDArray b, INDArray mean, INDArray var,
			List<Long> consumedInputs, Float epsilon, Boolean isTest, Float momentum, Boolean spatial) {
		if (isTest) {
			SameDiff sameDiff = DL4JSession.get();
			SDVariable batchNorm = sameDiff.nn.batchNorm(
					sameDiff.constant(x), 
					sameDiff.constant(mean), 
					sameDiff.constant(var), 
					sameDiff.constant(scale), 
					sameDiff.constant(b), 
					epsilon, 
					1);
			INDArray[] out = new INDArray[] { sameDiff.outputSingle(
					Collections.<String, INDArray>emptyMap(), 
					batchNorm.getVarName()) };
			return out;
		} else {
			throw new UnsupportedOperationException("[DL4JBatchNormalizationV1] Unable to handle isTest=false");
		}
	}

}
