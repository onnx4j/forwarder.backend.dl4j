package org.forwarder.backend.impls.dl4j.opsets.v5.ops;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

import org.forwarder.backend.impls.dl4j.opsets.v1.ops.DL4JReshapeV1;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.onnx4j.opsets.v5.ops.ReshapeV5;

import com.google.common.collect.Lists;

public class DL4JReshapeV5 extends DL4JReshapeV1 implements ReshapeV5<INDArray> {

	@Override
	public INDArray reshape(INDArray data, INDArray shape, List<Long> consumedInputs) {
		return super.reshape(data, Lists.newArrayList(this.calcNewShape(data, shape)), consumedInputs);
	}

	protected List<Long> calcNewShape(INDArray inputTensor, INDArray shapeTensor) {
		List<Long> newShape = new LinkedList<Long>();
		int autoExpandAxis = -1;
		for (int n = 0; n < shapeTensor.size(0); n++) {
			long len = shapeTensor.getLong(n);
			if (len > 0)
				newShape.add(n, len);
			else if (len == 0)
				newShape.add(n, inputTensor.shape()[n]);
			else if (len == -1) {
				if (autoExpandAxis != -1)
					throw new IllegalArgumentException(
							String.format("New shape can not hava more than one flag to execute auto-expand -> %s",
									Arrays.toString(inputTensor.shape())));

				autoExpandAxis = n;
				newShape.add(n, 1L);
			}
		}

		if (autoExpandAxis != -1) {
			//
			// do auto-expand
			//
			int sizeOfNewShape = 0;
			for (int n = 0; n < newShape.size(); n++) {
				if (sizeOfNewShape == 0)
					sizeOfNewShape += newShape.get(n);
				else
					sizeOfNewShape *= newShape.get(n);
			}
			int sizeOfData = 0;
			for (int n = 0; n < inputTensor.shape().length; n++) {
				if (sizeOfData == 0)
					sizeOfData += inputTensor.shape()[n];
				else
					sizeOfData *= inputTensor.shape()[n];
			}
			newShape.set(autoExpandAxis, (long) (sizeOfData / sizeOfNewShape));
		}

		return newShape;
	}

}
