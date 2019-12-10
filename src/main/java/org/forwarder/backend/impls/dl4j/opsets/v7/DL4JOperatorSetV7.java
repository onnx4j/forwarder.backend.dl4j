package org.forwarder.backend.impls.dl4j.opsets.v7;

import org.forwarder.backend.impls.dl4j.opsets.v6.DL4JOperatorSetV6;
import org.forwarder.backend.impls.dl4j.opsets.v7.ops.DL4JAveragePoolV7;
import org.forwarder.backend.impls.dl4j.opsets.v7.ops.DL4JBatchNormalizationV7;
import org.forwarder.backend.impls.dl4j.opsets.v7.ops.DL4JDropoutV7;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.onnx4j.opsets.v7.OperatorSetSpecV7;
import org.onnx4j.opsets.v7.ops.AveragePoolV7;
import org.onnx4j.opsets.v7.ops.BatchNormalizationV7;
import org.onnx4j.opsets.v7.ops.DropoutV7;

public class DL4JOperatorSetV7 extends DL4JOperatorSetV6 implements OperatorSetSpecV7<INDArray> {

	//@Override
	//public AcosV7<INDArray> getAcosV7() { return new DL4JAcosV7(); }

	@Override
	public BatchNormalizationV7<INDArray> getBatchNormalizationV7() { return new DL4JBatchNormalizationV7(); }

	@Override
	public DropoutV7<INDArray> getDropoutV7() { return new DL4JDropoutV7(); }

	@Override
	public AveragePoolV7<INDArray> getAveragePoolV7() { return new DL4JAveragePoolV7(); }

	public DL4JOperatorSetV7() {
		super(1, "", "", "", 7L, "ONNX OPSET-V7 USING DL4J BACKEND");
	}

	public DL4JOperatorSetV7(int irVersion, String irVersionPrerelease, String irBuildMetadata, String domain,
			long opsetVersion, String docString) {
		super(irVersion, irVersionPrerelease, irBuildMetadata, domain, opsetVersion, docString);
	}

}
