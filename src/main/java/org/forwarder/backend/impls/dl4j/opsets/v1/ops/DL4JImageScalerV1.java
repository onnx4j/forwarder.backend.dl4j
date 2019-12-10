package org.forwarder.backend.impls.dl4j.opsets.v1.ops;

import java.util.List;

import org.forwarder.backend.impls.dl4j.DL4JSession;
import org.forwarder.backend.impls.dl4j.opsets.DL4JOperator;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.guava.primitives.Floats;
import org.onnx4j.opsets.v1.ops.ImageScalerV1;

@Deprecated
public class DL4JImageScalerV1 extends DL4JOperator implements ImageScalerV1<INDArray> {

	@Override
	public INDArray scale(INDArray input, Float scale, List<Float> bias) {
		INDArray out = input.mul(scale);
		if (bias != null) {
			SameDiff sameDiff = DL4JSession.get();
			sameDiff.nn.biasAdd(sameDiff.constant(out), sameDiff.constant(Nd4j.create(Floats.toArray(bias))));
		}
		return out;
	}

}
