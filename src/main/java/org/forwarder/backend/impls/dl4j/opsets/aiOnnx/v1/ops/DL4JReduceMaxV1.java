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
import org.onnx4j.opsets.aiOnnx.v1.ops.ReduceMaxV1;

import com.google.common.primitives.Ints;

public class DL4JReduceMaxV1 extends DL4JAiOnnxOperator implements ReduceMaxV1<INDArray> {

	@Override
	public OperatorStatus getStatus() {
		return OperatorStatus.STABLE;
	}

	@Override
	public INDArray reduceMax(INDArray data, List<Long> axes, Long keepdims) {
		return data.max(keepdims == 1L ? true : false, Ints.toArray(axes));
	}

}