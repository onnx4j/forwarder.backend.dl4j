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
import org.nd4j.linalg.api.ops.impl.layers.convolution.Conv2D;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig.Conv2DConfigBuilder;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.factory.Nd4j;
import org.onnx4j.Inputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.ConvV1;
import org.onnx4j.opsets.operator.OperatorOutputs;

public class DL4JConvV1 extends DL4JAiOnnxOperator implements ConvV1 {

	@Override
	public OperatorOutputs<INDArray> forward(Node node, Inputs inputs) {
		ConvInputsV1<INDArray> castedOperatorInputs = new ConvInputsV1<INDArray>(node, inputs);
		INDArray x = castedOperatorInputs.getX();
		INDArray b = castedOperatorInputs.getB();
		INDArray w = castedOperatorInputs.getW();
		String autoPad = castedOperatorInputs.getAutoPad();
		List<Long> dilations = castedOperatorInputs.getDilations();
		Long group = castedOperatorInputs.getGroup();
		List<Long> kernelShape = castedOperatorInputs.getKernelShape();
		List<Long> pads = castedOperatorInputs.getPads();
		List<Long> strides = castedOperatorInputs.getStrides();
		return new ConvOutputV1<INDArray>(this.conv(x, w, b, autoPad, dilations, group, kernelShape, pads, strides));
	}

	protected INDArray conv(INDArray x, INDArray w, INDArray b, String autoPad, List<Long> dilations, Long group,
			List<Long> kernelShape, List<Long> pads, List<Long> strides) {
		Conv2DConfigBuilder configBuilder = Conv2DConfig.builder()
				.dataFormat(Conv2DConfig.NCHW)
				.dH(dilations.get(0))
				.dW(dilations.get(1))
				.sH(strides.get(0))
				.sW(strides.get(1))
				.kH(kernelShape.get(0))
				.kW(kernelShape.get(1))
				.isSameMode(autoPad.startsWith("SAME") ? true : false);

		if (pads != null && pads.size() >= 2) {
			configBuilder.pH(pads.get(0)).pW(pads.get(1));
		}
		
		Conv2D conv2d = new Conv2D(x, this.toHWCN(w), b, null, configBuilder.build());
		LongShapeDescriptor outputShapeDesc = conv2d.calculateOutputShape().get(0);
		INDArray out = Nd4j.create(outputShapeDesc);
		conv2d.addOutputArgument(out);
		return Nd4j.exec(conv2d)[0];
	}

	/**
	 * Translate weights from (M x C x KH x KW) to (KH x KW X C X M)
	 * 
	 * @param dataNCHW The data stored in NCHW mode
	 * @return
	 */
	private INDArray toHWCN(INDArray dataNCHW) {
		return dataNCHW.permute(2, 3, 1, 0);
	}

}