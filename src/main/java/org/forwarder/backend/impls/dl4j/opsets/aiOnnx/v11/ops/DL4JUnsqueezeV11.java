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
package org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v11.ops;

import java.util.List;

import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v1.ops.DL4JUnsqueezeV1;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.onnx4j.opsets.aiOnnx.v11.ops.UnsqueezeV11;

import com.google.common.primitives.Ints;

public class DL4JUnsqueezeV11 extends DL4JUnsqueezeV1 implements UnsqueezeV11<INDArray> {

	@Override
	public OperatorStatus getStatus() {
		return OperatorStatus.STABLE;
	}

	@Override
	public INDArray unsqueeze(INDArray data, List<Long> axes) {
		return this.unsqueeze(data, Ints.toArray(axes));
	}

	@Override
	protected INDArray unsqueeze(INDArray data, int[] axes) {
		for (int n = 0; n < axes.length; n++) {
			if (axes[n] < 0) {
				int adjustedAxis = Math.abs(axes[n]);
				if (adjustedAxis < 0 && adjustedAxis >= data.rank()) {
					throw new IllegalArgumentException(String.format(
							"The adjusted axis = %s, it must be a positive value and less than the rank of data(%s)",
							adjustedAxis, data.rank()));
				} else {
					axes[n] = adjustedAxis;
				}
			}
		}
		return super.unsqueeze(data, axes);
	}

}