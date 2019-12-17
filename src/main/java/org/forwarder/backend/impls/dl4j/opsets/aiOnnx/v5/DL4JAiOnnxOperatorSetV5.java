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
package org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v5;

import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v4.DL4JAiOnnxOperatorSetV4;
import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v5.ops.DL4JReshapeV5;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.onnx4j.opsets.aiOnnx.v5.AiOnnxOperatorSetSpecV5;
import org.onnx4j.opsets.aiOnnx.v5.ops.ReshapeV5;

public class DL4JAiOnnxOperatorSetV5 extends DL4JAiOnnxOperatorSetV4 implements AiOnnxOperatorSetSpecV5<INDArray> {

	@Override
	public ReshapeV5<INDArray> getReshapeV5() { return new DL4JReshapeV5(); }

	public DL4JAiOnnxOperatorSetV5() {
		this(1, "", "", 5L, "ONNX OPSET-V5 USING DL4J BACKEND");
	}

	public DL4JAiOnnxOperatorSetV5(int irVersion, String irVersionPrerelease, String irBuildMetadata,
			long opsetVersion, String docString) {
		super(irVersion, irVersionPrerelease, irBuildMetadata, opsetVersion, docString);
	}

}