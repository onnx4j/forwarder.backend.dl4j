package org.forwarder.backend.impls.dl4j.opsets.v5;

import org.forwarder.backend.impls.dl4j.opsets.v4.DL4JOperatorSetV4;
import org.forwarder.backend.impls.dl4j.opsets.v5.ops.DL4JReshapeV5;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.onnx4j.opsets.v5.OperatorSetSpecV5;
import org.onnx4j.opsets.v5.ops.ReshapeV5;

public class DL4JOperatorSetV5 extends DL4JOperatorSetV4 implements OperatorSetSpecV5<INDArray> {

	@Override
	public ReshapeV5<INDArray> getReshapeV5() { return new DL4JReshapeV5(); }

	public DL4JOperatorSetV5() {
		this(1, "", "", "", 5L, "ONNX OPSET-V5 USING DL4J BACKEND");
	}

	public DL4JOperatorSetV5(int irVersion, String irVersionPrerelease, String irBuildMetadata, String domain,
			long opsetVersion, String docString) {
		super(irVersion, irVersionPrerelease, irBuildMetadata, domain, opsetVersion, docString);
	}

}
