package org.forwarder.backend.impls.dl4j.opsets.v7.ops;

import java.util.List;

import org.forwarder.backend.impls.dl4j.opsets.v1.ops.DL4JAveragePoolV1;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.onnx4j.opsets.v7.ops.AveragePoolV7;

public class DL4JAveragePoolV7 extends DL4JAveragePoolV1 implements AveragePoolV7<INDArray> {

	@Override
	public INDArray averagePool(INDArray data, String autoPad, List<Long> kernelShape, List<Long> pads,
			List<Long> strides, Long countIncludePad) {
		if (countIncludePad != null && countIncludePad != 0L)
			throw new UnsupportedOperationException(
					String.format("[%s] Unable to handle \"countIncludePad\" is not equals to 0L", OP_TYPE));

		return super.averagePool(data, autoPad, kernelShape, pads, strides);
	}

}
