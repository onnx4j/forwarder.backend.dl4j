/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v1.ops;

import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.DL4JAiOnnxOperator;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.onnx4j.Inputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.IdentityV1;
import org.onnx4j.opsets.operator.OperatorOutputs;

public class DL4JIdentityV1 extends DL4JAiOnnxOperator implements IdentityV1 {

	private ActivationIdentity identity = new ActivationIdentity();

	@Override
	public OperatorOutputs<INDArray> forward(Node node, Inputs inputs) {
		IdentityInputsV1<INDArray> castedOperatorInputs = new IdentityInputsV1<INDArray>(node, inputs);
		INDArray input = castedOperatorInputs.getInput();
		return new IdentityOutputV1<INDArray>(this.identity(input));
	}

	protected INDArray identity(INDArray x0) {
		// TODO Auto-generated method stub
		return identity.getActivation(x0, false);
	}

}