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

import java.util.List;

import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.DL4JAiOnnxOperator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.onnx4j.Inputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.BatchNormalizationV1;
import org.onnx4j.opsets.operator.OperatorOutputs;

public class DL4JBatchNormalizationV1 extends DL4JAiOnnxOperator implements BatchNormalizationV1 {

	@Override
	public OperatorOutputs<INDArray> forward(Node node, Inputs inputs) {
		BatchNormalizationInputsV1<INDArray> castedOperatorInputs = new BatchNormalizationInputsV1<INDArray>(node, inputs);
		INDArray x = castedOperatorInputs.getX();
		INDArray scale = castedOperatorInputs.getScale();
		INDArray b = castedOperatorInputs.getB();
		INDArray mean = castedOperatorInputs.getMean();
		INDArray var = castedOperatorInputs.getVar();
		Float epsilon = castedOperatorInputs.getEpsilon();
		Boolean isTest = castedOperatorInputs.isTest();
		Float momentum = castedOperatorInputs.getMomentum();
		Boolean spatial = castedOperatorInputs.isSpatial();
		List<Long> consumedInputs = castedOperatorInputs.getConsumedInputs();
		return new BatchNormalizationOutputsV1<INDArray>(
				this.batchNormalization(x, scale, b, mean, var, consumedInputs, epsilon, isTest, momentum, spatial));
	}

	protected INDArray batchNormalization(INDArray x, INDArray scale, INDArray b, INDArray mean, INDArray var,
			List<Long> consumedInputs, Float epsilon, Boolean isTest, Float momentum, Boolean spatial) {
		if (isTest) {
			return Nd4j.nn.batchNorm(x, mean, var, scale, b, epsilon, 1);
			/*SameDiff sameDiff = DL4JSession.get();
			SDVariable batchNorm = sameDiff.nn.batchNorm(
					sameDiff.constant(x), 
					sameDiff.constant(mean), 
					sameDiff.constant(var), 
					sameDiff.constant(scale), 
					sameDiff.constant(b), 
					epsilon, 
					1);
			return sameDiff.outputSingle(
					Collections.<String, INDArray>emptyMap(), 
					batchNorm.getVarName());*/
		} else {
			throw new UnsupportedOperationException("[DL4JBatchNormalizationV1] Unable to handle isTest=false");
		}
	}

}