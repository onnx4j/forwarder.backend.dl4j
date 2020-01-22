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
package org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v1;

import org.forwarder.backend.impls.dl4j.DL4JBackend;
import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.DL4JAiOnnxOperatorSet;
import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v1.ops.DL4JAbsV1;
import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v1.ops.DL4JAddV1;
import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v1.ops.DL4JArgMaxV1;
import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v1.ops.DL4JAveragePoolV1;
import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v1.ops.DL4JBatchNormalizationV1;
import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v1.ops.DL4JCastV1;
import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v1.ops.DL4JConcatV1;
import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v1.ops.DL4JConstantV1;
import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v1.ops.DL4JConvV1;
import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v1.ops.DL4JDivV1;
import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v1.ops.DL4JDropoutV1;
import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v1.ops.DL4JIdentityV1;
import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v1.ops.DL4JImageScalerV1;
import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v1.ops.DL4JLeakyReluV1;
import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v1.ops.DL4JMatMulV1;
import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v1.ops.DL4JMaxPoolV1;
import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v1.ops.DL4JMulV1;
import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v1.ops.DL4JPadV1;
import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v1.ops.DL4JReduceMaxV1;
import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v1.ops.DL4JReluV1;
import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v1.ops.DL4JReshapeV1;
import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v1.ops.DL4JSigmoidV1;
import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v1.ops.DL4JSoftmaxV1;
import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v1.ops.DL4JSqueezeV1;
import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v1.ops.DL4JSubV1;
import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v1.ops.DL4JSumV1;
import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v1.ops.DL4JTransposeV1;
import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v1.ops.DL4JUnsqueezeV1;
import org.forwarder.opset.annotations.Opset;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.onnx4j.opsets.aiOnnx.v1.AiOnnxOperatorSetSpecV1;
import org.onnx4j.opsets.aiOnnx.v1.ops.AbsV1;
import org.onnx4j.opsets.aiOnnx.v1.ops.AddV1;
import org.onnx4j.opsets.aiOnnx.v1.ops.ArgMaxV1;
import org.onnx4j.opsets.aiOnnx.v1.ops.AveragePoolV1;
import org.onnx4j.opsets.aiOnnx.v1.ops.BatchNormalizationV1;
import org.onnx4j.opsets.aiOnnx.v1.ops.CastV1;
import org.onnx4j.opsets.aiOnnx.v1.ops.ConcatV1;
import org.onnx4j.opsets.aiOnnx.v1.ops.ConstantV1;
import org.onnx4j.opsets.aiOnnx.v1.ops.ConvV1;
import org.onnx4j.opsets.aiOnnx.v1.ops.DivV1;
import org.onnx4j.opsets.aiOnnx.v1.ops.DropoutV1;
import org.onnx4j.opsets.aiOnnx.v1.ops.GatherV1;
import org.onnx4j.opsets.aiOnnx.v1.ops.IdentityV1;
import org.onnx4j.opsets.aiOnnx.v1.ops.ImageScalerV1;
import org.onnx4j.opsets.aiOnnx.v1.ops.LeakyReluV1;
import org.onnx4j.opsets.aiOnnx.v1.ops.MatMulV1;
import org.onnx4j.opsets.aiOnnx.v1.ops.MaxPoolV1;
import org.onnx4j.opsets.aiOnnx.v1.ops.MulV1;
import org.onnx4j.opsets.aiOnnx.v1.ops.PadV1;
import org.onnx4j.opsets.aiOnnx.v1.ops.ReduceMaxV1;
import org.onnx4j.opsets.aiOnnx.v1.ops.ReluV1;
import org.onnx4j.opsets.aiOnnx.v1.ops.ReshapeV1;
import org.onnx4j.opsets.aiOnnx.v1.ops.SigmoidV1;
import org.onnx4j.opsets.aiOnnx.v1.ops.SoftmaxV1;
import org.onnx4j.opsets.aiOnnx.v1.ops.SqueezeV1;
import org.onnx4j.opsets.aiOnnx.v1.ops.SubV1;
import org.onnx4j.opsets.aiOnnx.v1.ops.SumV1;
import org.onnx4j.opsets.aiOnnx.v1.ops.TransposeV1;
import org.onnx4j.opsets.aiOnnx.v1.ops.UnsqueezeV1;

@Opset(backendName = DL4JBackend.BACKEND_NAME)
public class DL4JAiOnnxOperatorSetV1 extends DL4JAiOnnxOperatorSet implements AiOnnxOperatorSetSpecV1<INDArray> {

	@Override
	public AbsV1<INDArray> getAbsV1() { return new DL4JAbsV1(); }

	@Override
	public PadV1<INDArray> getPadV1() { return new DL4JPadV1(); }

	@Override
	public MatMulV1<INDArray> getMatMulV1() { return new DL4JMatMulV1(); }

	@Override
	public IdentityV1<INDArray> getIdentityV1() { return new DL4JIdentityV1(); }

	@Override
	public ArgMaxV1<INDArray> getArgMaxV1() { return new DL4JArgMaxV1(); }

	@Override
	public DivV1<INDArray> getDivV1() { return new DL4JDivV1(); }

	@Override
	public ReshapeV1<INDArray> getReshapeV1() { return new DL4JReshapeV1(); }

	@Override
	public AddV1<INDArray> getAddV1() { return new DL4JAddV1(); }

	@Override
	public MaxPoolV1<INDArray> getMaxPoolV1() { return new DL4JMaxPoolV1(); }

	@Override
	public ReluV1<INDArray> getReluV1() { return new DL4JReluV1(); }

	@Override
	public ConvV1<INDArray> getConvV1() { return new DL4JConvV1(); }

	@Override
	public ConstantV1<INDArray> getConstantV1() { return new DL4JConstantV1(); }

	@Override
	public ImageScalerV1<INDArray> getImageScalerV1() { return new DL4JImageScalerV1(); }

	@Override
	public BatchNormalizationV1<INDArray> getBatchNormalizationV1() { return new DL4JBatchNormalizationV1(); }

	@Override
	public LeakyReluV1<INDArray> getLeakyReluV1() { return new DL4JLeakyReluV1(); }

	@Override
	public ConcatV1<INDArray> getConcatV1() { return new DL4JConcatV1(); }

	@Override
	public MulV1<INDArray> getMulV1() { return new DL4JMulV1(); }

	@Override
	public DropoutV1<INDArray> getDropoutV1() { return new DL4JDropoutV1(); }

	@Override
	public AveragePoolV1<INDArray> getAveragePoolV1() { return new DL4JAveragePoolV1(); }

	@Override
	public CastV1<INDArray> getCastV1() { return new DL4JCastV1(); }
	
	// TODO: 
	@Override
	public GatherV1<INDArray> getGatherV1() { return null; }
	
	@Override
	public SubV1<INDArray> getSubV1() { return new DL4JSubV1(); }
	
	@Override
	public SumV1<INDArray> getSumV1() { return new DL4JSumV1(); }
	
	@Override
	public SigmoidV1<INDArray> getSigmoidV1() { return new DL4JSigmoidV1(); }
	
	@Override
	public SoftmaxV1<INDArray> getSoftmaxV1() { return new DL4JSoftmaxV1(); }

	@Override
	public SqueezeV1<INDArray> getSqueezeV1() { return new DL4JSqueezeV1(); }

	@Override
	public UnsqueezeV1<INDArray> getUnsqueezeV1() { return new DL4JUnsqueezeV1(); }

	@Override
	public ReduceMaxV1<INDArray> getReduceMaxV1() { return new DL4JReduceMaxV1(); }

	@Override
	public TransposeV1<INDArray> getTransposeV1() { return new DL4JTransposeV1(); }

	public DL4JAiOnnxOperatorSetV1() {
		this(1, "", "", 1L, "ONNX OPSET-V1 USING DL4J BACKEND");
	}

	public DL4JAiOnnxOperatorSetV1(int irVersion, String irVersionPrerelease, String irBuildMetadata,
			long opsetVersion, String docString) {
		super(irVersion, irVersionPrerelease, irBuildMetadata, opsetVersion, docString);
	}

}