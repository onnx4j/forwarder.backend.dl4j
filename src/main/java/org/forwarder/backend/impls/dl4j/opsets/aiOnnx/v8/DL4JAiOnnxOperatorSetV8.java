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
package org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v8;

import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v7.DL4JAiOnnxOperatorSetV7;
import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v8.ops.DL4JSumV8;
import org.onnx4j.opsets.domain.aiOnnx.v8.AiOnnxOpsetInitializerV8;
import org.onnx4j.opsets.domain.aiOnnx.v8.ops.SumV8;

public class DL4JAiOnnxOperatorSetV8 extends DL4JAiOnnxOperatorSetV7 implements AiOnnxOpsetInitializerV8 {

	@Override
	public SumV8 getSumV8() { return new DL4JSumV8(); }

	public DL4JAiOnnxOperatorSetV8() {
		super(1, "", "", 8L, "ONNX OPSET-V8 USING DL4J BACKEND");
	}

	public DL4JAiOnnxOperatorSetV8(int irVersion, String irVersionPrerelease, String irBuildMetadata,
			long opsetVersion, String docString) {
		super(irVersion, irVersionPrerelease, irBuildMetadata, opsetVersion, docString);
	}

}