package org.forwarder.backend.impls.dl4j.opsets.v8;

import org.forwarder.backend.impls.dl4j.opsets.v7.DL4JOperatorSetV7;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.onnx4j.opsets.v8.OperatorSetSpecV8;

public class DL4JOperatorSetV8 extends DL4JOperatorSetV7 implements OperatorSetSpecV8<INDArray> {

	public DL4JOperatorSetV8() {
		super(1, "", "", "", 8L, "ONNX OPSET-V8 USING DL4J BACKEND");
	}

	public DL4JOperatorSetV8(int irVersion, String irVersionPrerelease, String irBuildMetadata, String domain,
			long opsetVersion, String docString) {
		super(irVersion, irVersionPrerelease, irBuildMetadata, domain, opsetVersion, docString);
	}

}
