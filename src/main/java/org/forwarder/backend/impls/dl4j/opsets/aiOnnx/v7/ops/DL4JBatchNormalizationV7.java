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
package org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v7.ops;

import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v6.ops.DL4JBatchNormalizationV6;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.onnx4j.Inputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.opsets.domain.aiOnnx.v7.ops.BatchNormalizationV7;
import org.onnx4j.opsets.operator.OperatorOutputs;

public class DL4JBatchNormalizationV7 extends DL4JBatchNormalizationV6 implements BatchNormalizationV7 {

	@Override
	public OperatorOutputs<INDArray> forward(Node node, Inputs inputs) {
		BatchNormalizationInputsV6<INDArray> castedOperatorInputs = new BatchNormalizationInputsV7<INDArray>(node, inputs);
		INDArray x = castedOperatorInputs.getX();
		INDArray scale = castedOperatorInputs.getScale();
		INDArray b = castedOperatorInputs.getB();
		INDArray mean = castedOperatorInputs.getMean();
		INDArray var = castedOperatorInputs.getVar();
		Float epsilon = castedOperatorInputs.getEpsilon();
		Float momentum = castedOperatorInputs.getMomentum();
		Boolean spatial = castedOperatorInputs.isSpatial();
		return new BatchNormalizationOutputsV7<INDArray>(
				this.batchNormalization(x, scale, b, mean, var, null, epsilon, true, momentum, spatial));
	}

}