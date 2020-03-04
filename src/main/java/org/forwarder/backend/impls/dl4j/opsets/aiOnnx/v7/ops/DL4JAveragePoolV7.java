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

import java.util.List;

import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v1.ops.DL4JAveragePoolV1;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.onnx4j.Inputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.opsets.domain.aiOnnx.v7.ops.AveragePoolV7;
import org.onnx4j.opsets.operator.OperatorOutputs;

public class DL4JAveragePoolV7 extends DL4JAveragePoolV1 implements AveragePoolV7 {

	@Override
	public OperatorOutputs<INDArray> forward(Node node, Inputs inputs) {
		AveragePoolInputsV7<INDArray> castedOperatorInputs = new AveragePoolInputsV7<INDArray>(node, inputs);
		INDArray x = castedOperatorInputs.getX();
		String autoPad = castedOperatorInputs.getAutoPad();
		List<Long> kernelShape = castedOperatorInputs.getKernelShape();
		List<Long> pads = castedOperatorInputs.getPads();
		List<Long> strides = castedOperatorInputs.getStrides();
		return new AveragePoolOutputV7<INDArray>(this.averagePool(x, autoPad, kernelShape, pads, strides));
	}

	protected INDArray averagePool(INDArray data, String autoPad, List<Long> kernelShape, List<Long> pads,
			List<Long> strides, Long countIncludePad) {
		if (countIncludePad != null && countIncludePad != 0L)
			throw new UnsupportedOperationException(
					String.format("[%s] Unable to handle \"countIncludePad\" is not equals to 0L", OP_TYPE));

		return super.averagePool(data, autoPad, kernelShape, pads, strides);
	}

}