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
package org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v6;

import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v5.DL4JAiOnnxOperatorSetV5;
import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v6.ops.DL4JCastV6;
import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v6.ops.DL4JDropoutV6;
import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v6.ops.DL4JMulV6;
import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v6.ops.DL4JSigmoidV6;
import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v6.ops.DL4JSubV6;
import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v6.ops.DL4JSumV6;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.onnx4j.opsets.aiOnnx.v6.AiOnnxOperatorSetSpecV6;
import org.onnx4j.opsets.aiOnnx.v6.ops.CastV6;
import org.onnx4j.opsets.aiOnnx.v6.ops.DropoutV6;
import org.onnx4j.opsets.aiOnnx.v6.ops.MulV6;
import org.onnx4j.opsets.aiOnnx.v6.ops.SigmoidV6;
import org.onnx4j.opsets.aiOnnx.v6.ops.SubV6;
import org.onnx4j.opsets.aiOnnx.v6.ops.SumV6;

public class DL4JAiOnnxOperatorSetV6 extends DL4JAiOnnxOperatorSetV5 implements AiOnnxOperatorSetSpecV6<INDArray> {

	@Override
	public MulV6<INDArray> getMulV6() { return new DL4JMulV6(); }

	@Override
	public DropoutV6<INDArray> getDropoutV6() { return new DL4JDropoutV6(); }

	@Override
	public CastV6<INDArray> getCastV6() { return new DL4JCastV6(); }

	@Override
	public SubV6<INDArray> getSubV6() { return new DL4JSubV6(); }

	@Override
	public SumV6<INDArray> getSumV6() { return new DL4JSumV6(); }
	
	@Override
	public SigmoidV6<INDArray> getSigmoidV6() { return new DL4JSigmoidV6(); }

	public DL4JAiOnnxOperatorSetV6() {
		this(1, "", "", 6L, "ONNX OPSET-V6 USING DL4J BACKEND");
	}

	public DL4JAiOnnxOperatorSetV6(int irVersion, String irVersionPrerelease, String irBuildMetadata,
			long opsetVersion, String docString) {
		super(irVersion, irVersionPrerelease, irBuildMetadata, opsetVersion, docString);
	}

}