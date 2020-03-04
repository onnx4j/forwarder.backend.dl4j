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
package org.forwarder.backend.impls.dl4j;

import java.util.Arrays;

import org.forwarder.Backend;
import org.forwarder.Model;
import org.forwarder.Session;
import org.forwarder.backend.impls.dl4j.utils.DL4JDataTypeHelper;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.onnx4j.Tensor;
import org.onnx4j.TensorManager;
import org.onnx4j.tensor.Shape;
import org.onnx4j.tensor.TensorBuilder;

public class DL4JBackend extends Backend<INDArray> {

	public static final String BACKEND_NAME = "DL4J";

	public DL4JBackend() {
		super();
	}

	public DL4JBackend(Model model) {
		super(model);
	}

	@Override
	public String getName() {
		return BACKEND_NAME;
	}

	@Override
	public void disposeBackendTensor(INDArray backendTensor) {
		backendTensor.close();
	}

	@Override
	public INDArray toBackendTensor(TensorManager<INDArray> tensorManager, org.onnx4j.Tensor onnx4jTensor) {
		DataType backendDataType = DL4JDataTypeHelper.toDl4jDataType(onnx4jTensor.getDataType());
		DataBuffer dataBuffer = Nd4j.createBuffer(onnx4jTensor.getData(), backendDataType, (int) onnx4jTensor.getElementSize());
		int shape[] = Arrays.stream(onnx4jTensor.getShape()).mapToInt(i -> (int) i).toArray();
		INDArray ndArray = Nd4j.create(dataBuffer, shape);
		
		//
		// Attach to Onnx4j.TensorManager if the backend tensor had not been attached.
		//
		if (ndArray.isAttached() == false)
			tensorManager.attach(onnx4jTensor.getName(), ndArray);
		
		return ndArray;
	}

	@Override
	public org.onnx4j.Tensor toNativeTensor(TensorManager<Tensor> tensorManager, String name, INDArray backendTensor) {
		org.onnx4j.tensor.DataType onnx4jDataType = DL4JDataTypeHelper.toOnnx4jDataType(backendTensor.data().dataType());
		TensorBuilder builder = TensorBuilder
				.builder(onnx4jDataType, Shape.create(backendTensor.shape()), backendTensor.data().asNio())
				.name(name)
				.docString("Created from org.nd4j.linalg.api.ndarray.INDArray in DL4JBackend.toTensor()");
		//
		// Attach to Onnx4j.TensorManager if the backend tensor had not been attached.
		//
		if (backendTensor.isAttached() == false)
			builder.manager(tensorManager);
		
		return builder.build();
	}

	@Override
	public Session<INDArray> newSession() {
		return new DL4JSession(this);
	}

}