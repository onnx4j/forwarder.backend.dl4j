package org.forwarder.backend.impls.dl4j.opsets.v2;

import org.forwarder.backend.impls.dl4j.opsets.v1.DL4JOperatorSetV1;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.onnx4j.opsets.v2.OperatorSetSpecV2;

public class DL4JOperatorSetV2 extends DL4JOperatorSetV1 implements OperatorSetSpecV2<INDArray> {

	//@Override
	//public PadV2<INDArray> getPadV2() { return new DL4JPadV2(); }

	public DL4JOperatorSetV2() {
		this(1, "", "", "", 2L, "ONNX OPSET-V2 USING DL4J BACKEND");
	}

	public DL4JOperatorSetV2(int irVersion, String irVersionPrerelease, String irBuildMetadata, String domain,
			long opsetVersion, String docString) {
		super(irVersion, irVersionPrerelease, irBuildMetadata, domain, opsetVersion, docString);
	}

}
