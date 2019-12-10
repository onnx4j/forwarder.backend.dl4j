package org.forwarder.backend.impls.dl4j.opsets.v1.ops;

import java.util.List;

import org.forwarder.backend.impls.dl4j.opsets.DL4JOperator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.shade.guava.primitives.Longs;
import org.onnx4j.opsets.v1.ops.ReshapeV1;

public class DL4JReshapeV1 extends DL4JOperator implements ReshapeV1<INDArray> {

	@Override
	public INDArray reshape(INDArray data, List<Long> shape, List<Long> consumedInputs) {
		return data.reshape(Longs.toArray(shape));
	}

}
