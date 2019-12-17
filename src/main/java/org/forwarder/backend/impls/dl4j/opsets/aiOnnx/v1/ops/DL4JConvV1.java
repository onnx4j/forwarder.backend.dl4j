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

import java.util.Collections;
import java.util.List;

import org.forwarder.backend.impls.dl4j.DL4JSession;
import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.DL4JAiOnnxOperator;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig.Conv2DConfigBuilder;
import org.onnx4j.opsets.aiOnnx.v1.ops.ConvV1;

public class DL4JConvV1 extends DL4JAiOnnxOperator implements ConvV1<INDArray> {

	@Override
	public INDArray conv(INDArray x, INDArray w, INDArray b, String autoPad, List<Long> dilations, Long group,
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

		SameDiff sameDiff = DL4JSession.get();
		SDVariable conv2d;
		if (b == null) {
			conv2d = sameDiff.cnn.conv2d(
					sameDiff.constant(x),
					sameDiff.constant(this.toHWCN(w)),
					configBuilder.build());
		} else {
			conv2d = sameDiff.cnn.conv2d(
					sameDiff.constant(x),
					sameDiff.constant(this.toHWCN(w)),
					sameDiff.constant(b),
					configBuilder.build());
		}
		
		return sameDiff.outputSingle(
				Collections.<String, INDArray>emptyMap(), 
				conv2d.getVarName());
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