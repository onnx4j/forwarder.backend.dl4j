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
package org.forwarder.backend.impls.dl4j.opsets;

import java.util.Arrays;

import org.forwarder.opset.operator.Executable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.onnx4j.opsets.operator.Field.TypeConstraint;
import org.onnx4j.opsets.operator.OperatorInputs;
import org.onnx4j.opsets.operator.fields.InputField;
import org.onnx4j.tensor.DataType;

public abstract class DL4JOperator implements Executable<INDArray> {
	
	public void preconditions(OperatorInputs<INDArray> operatorInputs) {
		for (InputField<INDArray> inputField : operatorInputs.getInputFields()) {
			TypeConstraint constraints = inputField.getConstraints();
			/*Arrays.asList(constraints.getDataTypes()).contains(arg0)
			for (DataType availableType : constraints.getDataTypes()) {
				Arrays.
			}*/
		}
	}

}