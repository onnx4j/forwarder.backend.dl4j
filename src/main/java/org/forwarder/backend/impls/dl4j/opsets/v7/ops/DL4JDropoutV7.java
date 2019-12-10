package org.forwarder.backend.impls.dl4j.opsets.v7.ops;

import java.util.List;

import org.forwarder.backend.impls.dl4j.opsets.v6.ops.DL4JDropoutV6;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.onnx4j.opsets.v7.ops.DropoutV7;

public class DL4JDropoutV7 extends DL4JDropoutV6 implements DropoutV7<INDArray> {

	@Override
	public List<INDArray> dropout(INDArray data, Float ratio) {
		return super.dropout(data, true, ratio);
	}

}
