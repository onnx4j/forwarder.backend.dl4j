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
package org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v11;

import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v10.DL4JAiOnnxOperatorSetV10;
import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v11.ops.DL4JReduceMaxV11;
import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v11.ops.DL4JSoftmaxV11;
import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v11.ops.DL4JSqueezeV11;
import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v11.ops.DL4JUnsqueezeV11;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.onnx4j.opsets.aiOnnx.v11.AiOnnxOperatorSetSpecV11;
import org.onnx4j.opsets.aiOnnx.v11.ops.ReduceMaxV11;
import org.onnx4j.opsets.aiOnnx.v11.ops.SoftmaxV11;
import org.onnx4j.opsets.aiOnnx.v11.ops.SqueezeV11;
import org.onnx4j.opsets.aiOnnx.v11.ops.UnsqueezeV11;

public class DL4JAiOnnxOperatorSetV11 extends DL4JAiOnnxOperatorSetV10 implements AiOnnxOperatorSetSpecV11<INDArray> {

	@Override
	public SoftmaxV11<INDArray> getSoftmaxV11() { return new DL4JSoftmaxV11(); }

	@Override
	public SqueezeV11<INDArray> getSqueezeV11() { return new DL4JSqueezeV11(); }

	@Override
	public UnsqueezeV11<INDArray> getUnsqueezeV11() { return new DL4JUnsqueezeV11(); }

	@Override
	public ReduceMaxV11<INDArray> getReduceMaxV11() { return new DL4JReduceMaxV11(); }

	public DL4JAiOnnxOperatorSetV11() {
		super(1, "", "", 11L, "ONNX OPSET-V11 USING DL4J BACKEND");
	}

	public DL4JAiOnnxOperatorSetV11(int irVersion, String irVersionPrerelease, String irBuildMetadata,
			long opsetVersion, String docString) {
		super(irVersion, irVersionPrerelease, irBuildMetadata, opsetVersion, docString);
	}

}