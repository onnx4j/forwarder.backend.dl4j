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
import org.nd4j.linalg.api.ops.impl.layers.convolution.AvgPooling2D;
import org.nd4j.linalg.api.ops.impl.layers.convolution.Pooling2D.Pooling2DType;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Pooling2DConfig;
import org.nd4j.linalg.factory.Nd4j;
import org.onnx4j.Inputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.AveragePoolV1;
import org.onnx4j.opsets.operator.OperatorOutputs;

public class DL4JAveragePoolV1 extends DL4JAiOnnxOperator implements AveragePoolV1 {

	@Override
	public OperatorOutputs<INDArray> forward(Node node, Inputs inputs) {
		AveragePoolInputsV1<INDArray> castedOperatorInputs = new AveragePoolInputsV1<INDArray>(node, inputs);
		INDArray x = castedOperatorInputs.getX();
		String autoPad = castedOperatorInputs.getAutoPad();
		List<Long> kernelShape = castedOperatorInputs.getKernelShape();
		List<Long> pads = castedOperatorInputs.getPads();
		List<Long> strides = castedOperatorInputs.getStrides();
		return new AveragePoolOutputV1<INDArray>(this.averagePool(x, autoPad, kernelShape, pads, strides));
	}

	protected INDArray averagePool(INDArray data, String autoPad, List<Long> kernelShape, List<Long> pads,
			List<Long> strides) {
		Pooling2DConfig pooling2DConfig = Pooling2DConfig.builder()
				.isNHWC(false)
				.sH(strides.get(0))
				.sW(strides.get(1))
				.kH(kernelShape.get(0))
				.kW(kernelShape.get(1))
				.type(Pooling2DType.MAX)
				.isSameMode(autoPad.startsWith("SAME") ? true : false)
				.build();
		AvgPooling2D avgPooling2d = new AvgPooling2D(data, null, pooling2DConfig);
		return Nd4j.exec(avgPooling2d)[0];
	}

}