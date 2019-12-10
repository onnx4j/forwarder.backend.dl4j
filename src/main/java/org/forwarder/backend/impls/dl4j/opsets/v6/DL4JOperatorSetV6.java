package org.forwarder.backend.impls.dl4j.opsets.v6;

import org.forwarder.backend.impls.dl4j.opsets.v5.DL4JOperatorSetV5;
import org.forwarder.backend.impls.dl4j.opsets.v6.ops.DL4JDropoutV6;
import org.forwarder.backend.impls.dl4j.opsets.v6.ops.DL4JMulV6;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.onnx4j.opsets.v6.OperatorSetSpecV6;
import org.onnx4j.opsets.v6.ops.DropoutV6;
import org.onnx4j.opsets.v6.ops.MulV6;

public class DL4JOperatorSetV6 extends DL4JOperatorSetV5 implements OperatorSetSpecV6<INDArray> {

	@Override
	public MulV6<INDArray> getMulV6() { return new DL4JMulV6(); }

	@Override
	public DropoutV6<INDArray> getDropoutV6() { return new DL4JDropoutV6(); }

	public DL4JOperatorSetV6() {
		this(1, "", "", "", 6L, "ONNX OPSET-V6 USING DL4J BACKEND");
	}

	public DL4JOperatorSetV6(int irVersion, String irVersionPrerelease, String irBuildMetadata, String domain,
			long opsetVersion, String docString) {
		super(irVersion, irVersionPrerelease, irBuildMetadata, domain, opsetVersion, docString);
	}

}
