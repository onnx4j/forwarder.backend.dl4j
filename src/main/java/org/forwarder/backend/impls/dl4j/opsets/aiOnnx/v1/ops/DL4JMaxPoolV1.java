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
import org.nd4j.linalg.api.ops.impl.layers.convolution.Pooling2D.Pooling2DType;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Pooling2DConfig;
import org.onnx4j.opsets.aiOnnx.v1.ops.MaxPoolV1;

public class DL4JMaxPoolV1 extends DL4JAiOnnxOperator implements MaxPoolV1<INDArray> {

	@Override
	public INDArray maxpool(INDArray data, String autoPad, List<Long> kernelShape, List<Long> pads,
			List<Long> strides) {
		SameDiff sameDiff = DL4JSession.get();
		
		Pooling2DConfig pooling2DConfig = Pooling2DConfig.builder()
				.isNHWC(false)
				.sH(strides.get(0))
				.sW(strides.get(1))
				.kH(kernelShape.get(0))
				.kW(kernelShape.get(1))
				.type(Pooling2DType.MAX)
				.isSameMode(autoPad.startsWith("SAME") ? true : false)
				.build();
		
		SDVariable maxPooling2d = sameDiff.cnn.maxPooling2d(sameDiff.constant(data), pooling2DConfig);
		
		return sameDiff.outputSingle(
				Collections.<String, INDArray>emptyMap(), 
				maxPooling2d.getVarName());
	}

}