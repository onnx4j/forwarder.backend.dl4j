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

import org.forwarder.backend.impls.dl4j.DL4JSession;
import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.DL4JAiOnnxOperator;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.shape.Squeeze;
import org.onnx4j.opsets.aiOnnx.v1.ops.SqueezeV1;

import com.google.common.primitives.Ints;

public class DL4JSqueezeV1 extends DL4JAiOnnxOperator implements SqueezeV1<INDArray> {
	
	@Override
	public OperatorStatus getStatus() {
		return OperatorStatus.STABLE;
	}

	@Override
	public INDArray squeeze(INDArray data, List<Long> axes) {
		long[] shape = data.shape();
		if (axes.size() == 0) {
			//
			// If axes is not provided, all the single dimensions will be
			// removed
			// from the shape.
			//
			for (int n = 0; n < shape.length; n++) {
				if (shape[n] == 1) {
					axes.add((long) n);
				}
			}
		} else {
			//
			// If an axis is selected with shape entry not equal to one, an
			// error is raised.
			//
			for (Long axis : axes) {
				if (shape[axis.intValue()] != 1) {
					throw new IllegalArgumentException(String.format(
							"An axis(%s) is selected with shape entry expects to equal to 1, but is equal to %s", axis,
							shape[axis.intValue()]));
				}
			}
		}

		SameDiff sameDiff = DL4JSession.get();
		Squeeze squeeze = new Squeeze(sameDiff, sameDiff.constant(data), Ints.toArray(axes));
		SDVariable out = squeeze.outputVariable();
		/*
		 * return sameDiff.outputSingle( Collections.<String,
		 * INDArray>emptyMap(), out.getVarName());
		 */
		return out.eval();
	}

}