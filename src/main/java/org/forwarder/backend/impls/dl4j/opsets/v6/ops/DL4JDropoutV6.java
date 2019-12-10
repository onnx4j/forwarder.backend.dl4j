package org.forwarder.backend.impls.dl4j.opsets.v6.ops;

import java.util.List;

import org.forwarder.backend.impls.dl4j.opsets.v1.ops.DL4JDropoutV1;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.onnx4j.opsets.v6.ops.DropoutV6;

public class DL4JDropoutV6 extends DL4JDropoutV1 implements DropoutV6<INDArray> {

	@Override
	public List<INDArray> dropout(INDArray data, Boolean isTest, Float ratio) {
		return super.dropout(data, isTest, ratio, null);
	}

}
