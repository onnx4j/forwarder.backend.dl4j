package org.forwarder.backend.impls.dl4j.opsets;

import org.forwarder.backend.impls.dl4j.DL4JBackend;
import org.forwarder.opset.annotations.Opset;
import org.onnx4j.opsets.OperatorSet;

@Opset(backendName = DL4JBackend.BACKEND_NAME)
public abstract class DL4JOperatorSet extends OperatorSet {

	public DL4JOperatorSet(int irVersion, String irVersionPrerelease, String irBuildMetadata, String domain,
			long opsetVersion, String docString) {
		super(irVersion, irVersionPrerelease, irBuildMetadata, domain, opsetVersion, docString);
		// TODO Auto-generated constructor stub
	}

}
