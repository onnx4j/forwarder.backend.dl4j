package org.forwarder.backend.impls.dl4j.opsets.v4;

import org.forwarder.backend.impls.dl4j.opsets.v2.DL4JOperatorSetV2;
import org.forwarder.backend.impls.dl4j.opsets.v4.ops.DL4JConcatV4;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.onnx4j.opsets.v4.OperatorSetSpecV4;
import org.onnx4j.opsets.v4.ops.ConcatV4;

public class DL4JOperatorSetV4 extends DL4JOperatorSetV2 implements OperatorSetSpecV4<INDArray> {

	@Override
	public ConcatV4<INDArray> getConcatV4() { return new DL4JConcatV4(); }

	public DL4JOperatorSetV4() {
		this(1, "", "", "", 4L, "ONNX OPSET-V4 USING TENSORFLOW BACKEND");
	}

	public DL4JOperatorSetV4(int irVersion, String irVersionPrerelease, String irBuildMetadata, String domain,
			long opsetVersion, String docString) {
		super(irVersion, irVersionPrerelease, irBuildMetadata, domain, opsetVersion, docString);
	}

}
