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
package org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v6.ops;

import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v1.ops.DL4JSubV1;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.onnx4j.Inputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.opsets.domain.aiOnnx.v6.ops.SubV6;
import org.onnx4j.opsets.operator.OperatorOutputs;

public class DL4JSubV6 extends DL4JSubV1 implements SubV6 {

	@Override
	public OperatorOutputs<INDArray> forward(Node node, Inputs inputs) {
		SubInputsV6<INDArray> castedOperatorInputs = new SubInputsV6<INDArray>(node, inputs);
		INDArray a = castedOperatorInputs.getA();
		INDArray b = castedOperatorInputs.getB();
		Long axis = castedOperatorInputs.getAxis();
		Long broadcast = castedOperatorInputs.getBroadcast();
		return new SubOutputV6<INDArray>(this.sub(a, b, axis, broadcast));
	}

	protected INDArray sub(INDArray a, INDArray b, Long axis, Long broadcast) {
		return super.sub(a, b, axis, broadcast, null);
	}

}