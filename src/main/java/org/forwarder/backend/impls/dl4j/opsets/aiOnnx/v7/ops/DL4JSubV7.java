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

import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v1.ops.DL4JSubV1;
import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v6.ops.DL4JSubV6;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Broadcast;
import org.onnx4j.opsets.aiOnnx.v6.ops.SubV6;
import org.onnx4j.opsets.aiOnnx.v7.ops.SubV7;

public class DL4JSubV7 extends DL4JSubV6 implements SubV7<INDArray> {

	@Override
	public INDArray sub(INDArray a, INDArray b) {
		return a.sub(b);
	}

}