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
package org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v7;

import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v6.DL4JAiOnnxOperatorSetV6;
import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v7.ops.DL4JAveragePoolV7;
import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v7.ops.DL4JBatchNormalizationV7;
import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v7.ops.DL4JDropoutV7;
import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v7.ops.DL4JSubV7;
import org.onnx4j.opsets.domain.aiOnnx.v7.AiOnnxOperatorSetInitializerV7;
import org.onnx4j.opsets.domain.aiOnnx.v7.ops.AveragePoolV7;
import org.onnx4j.opsets.domain.aiOnnx.v7.ops.BatchNormalizationV7;
import org.onnx4j.opsets.domain.aiOnnx.v7.ops.DropoutV7;
import org.onnx4j.opsets.domain.aiOnnx.v7.ops.SubV7;

public class DL4JAiOnnxOperatorSetV7 extends DL4JAiOnnxOperatorSetV6 implements AiOnnxOperatorSetInitializerV7 {

	//@Override
	//public AcosV7 getAcosV7() { return new DL4JAcosV7(); }

	@Override
	public BatchNormalizationV7 getBatchNormalizationV7() { return new DL4JBatchNormalizationV7(); }

	@Override
	public DropoutV7 getDropoutV7() { return new DL4JDropoutV7(); }

	@Override
	public AveragePoolV7 getAveragePoolV7() { return new DL4JAveragePoolV7(); }

	@Override
	public SubV7 getSubV7() { return new DL4JSubV7(); }

	public DL4JAiOnnxOperatorSetV7() {
		super(1, "", "", 7L, "ONNX OPSET-V7 USING DL4J BACKEND");
	}

	public DL4JAiOnnxOperatorSetV7(int irVersion, String irVersionPrerelease, String irBuildMetadata,
			long opsetVersion, String docString) {
		super(irVersion, irVersionPrerelease, irBuildMetadata, opsetVersion, docString);
	}

}