package org.forwarder.backend.impls.dl4j.opsets.v1.ops;

import org.forwarder.backend.impls.dl4j.opsets.DL4JOperator;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.onnx4j.opsets.v1.ops.IdentityV1;

public class DL4JIdentityV1 extends DL4JOperator implements IdentityV1<INDArray> {
	
	private ActivationIdentity identity = new ActivationIdentity();

	@Override
	public INDArray identity(INDArray x0) {
		// TODO Auto-generated method stub
		return identity.getActivation(x0, false);
	}

}
