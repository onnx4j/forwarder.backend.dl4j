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
import org.onnx4j.opsets.aiOnnx.v1.ops.ConcatV1;

public class DL4JConcatV1 extends DL4JAiOnnxOperator implements ConcatV1<INDArray> {

	@Override
	public INDArray concat(List<INDArray> inputs, Long axis) {
		SameDiff sameDiff = DL4JSession.get();

		SDVariable[] sdInputs = new SDVariable[inputs.size()];
		for (int n = 0; n < sdInputs.length; n++) {
			sdInputs[n] = sameDiff.constant(inputs.get(n));
		}

		SDVariable concat = sameDiff.concat(axis.intValue(), sdInputs);
		return sameDiff.outputSingle(Collections.<String, INDArray>emptyMap(), concat.getVarName());
	}

}