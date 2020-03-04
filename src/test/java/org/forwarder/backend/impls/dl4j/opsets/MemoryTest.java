package org.forwarder.backend.impls.dl4j.opsets;

import java.util.Random;

import org.bytedeco.javacpp.Pointer;
import org.junit.Test;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.api.memory.enums.MemoryKind;
import org.nd4j.linalg.api.memory.enums.MirroringPolicy;
import org.nd4j.linalg.api.memory.enums.SpillPolicy;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.memory.abstracts.Nd4jWorkspace;

public class MemoryTest {

	@Test
	public void testMemory() {
		System.out.println("totalPhysicalBytes: " + Pointer.maxBytes() / 1024 / 1024 / 1024 + "G");

		WorkspaceConfiguration wsCfg = WorkspaceConfiguration.builder().initialSize(10 * 1024 * 1024)
				.maxSize(10 * 1024 * 1024).overallocationLimit(0.1).policyAllocation(AllocationPolicy.STRICT)
				.policyLearning(LearningPolicy.FIRST_LOOP).policyMirroring(MirroringPolicy.FULL)
				.policySpill(SpillPolicy.EXTERNAL).build();

		int len = 10000000;

		for (int n = 0; n < 1000000; n++) {
			// try (Nd4jWorkspace ws = (Nd4jWorkspace)
			// Nd4j.getWorkspaceManager().getAndActivateWorkspace(wsCfg,
			// "testws")) {
			// ws.enableDebug(true);
			//System.out.println("MMUL" + Nd4j.linspace(1, len, len).reshape(1, len)
			//		.mmul(Nd4j.linspace(1, len, len).reshape(1, len).transpose()));
			// }
			/*Nd4j.linspace(1, len, len).reshape(1, len)
			.mmul(Nd4j.linspace(1, len, len).reshape(1, len).transpose());
			System.gc();*/
			INDArray a = Nd4j.linspace(1, len, len);
			INDArray b = a.reshape(1, len);
			INDArray c = Nd4j.linspace(1, len, len);
			INDArray d = a.reshape(1, len);
			INDArray e = d.transpose();
			INDArray f = b.mmul(e);
			System.out.println(f);
			//f.close();
			//e.close();
			//d.close();
			c.close();
			//b.close();
			a.close();
		}
	}

	@Test
	public void testLinearRegression() {
		int len = 100000;
		int iters = 100000;
		int exampleCount = 100000;
		double learningRate = 0.01;
		Random random = new Random();
		double[] data = new double[exampleCount * 3];
		double[] param = new double[exampleCount * 3];
		for (int i = 0; i < exampleCount * 3; i++) {
			data[i] = random.nextDouble();
		}
		for (int i = 0; i < exampleCount * 3; i++) {
			param[i] = 3;
			param[++i] = 4;
			param[++i] = 5;
		}
		INDArray features = Nd4j.create(data, new int[] { exampleCount, 3 });
		INDArray params = Nd4j.create(param, new int[] { exampleCount, 3 });
		INDArray label = features.mul(params).sum(1).add(10);
		double[] parameter = new double[] { 1.0, 1.0, 1.0, 1.0 };
		long startTime = System.currentTimeMillis();
		for (int i = 0; i < iters; i++) {
			BGD(features, label, learningRate, parameter);
			System.out.println("MMUL" + Nd4j.linspace(1, len, len).reshape(1, len)
					.mmul(Nd4j.linspace(1, len, len).reshape(1, len).transpose()));
			System.gc();
		}
		System.out.println("耗时：" + (System.currentTimeMillis() - startTime));
	}

	private static void BGD(INDArray features, INDArray label, double learningRate, double[] parameter) {
		INDArray temp = features.getColumn(0).mul(parameter[0]).add(features.getColumn(1).mul(parameter[1]))
				.add(features.getColumn(2).mul(parameter[2])).add(parameter[3]).sub(label);
		parameter[0] = parameter[0]
				- 2 * learningRate * temp.mul(features.getColumn(0)).sum(0).getDouble(0) / features.size(0);
		parameter[1] = parameter[1]
				- 2 * learningRate * temp.mul(features.getColumn(1)).sum(0).getDouble(0) / features.size(0);
		parameter[2] = parameter[2]
				- 2 * learningRate * temp.mul(features.getColumn(2)).sum(0).getDouble(0) / features.size(0);
		parameter[3] = parameter[3] - 2 * learningRate * temp.sum(0).getDouble(0) / features.size(0);
		INDArray functionResult = features.getColumn(0).mul(parameter[0]).add(features.getColumn(1).mul(parameter[1]))
				.add(features.getColumn(2).mul(parameter[2])).add(parameter[3]).sub(label);// 用最新的参数计算总损失用
		double totalLoss = functionResult.mul(functionResult).sum(0).getDouble(0);
		System.out.println("totalLoss:" + totalLoss);
		System.out.println(parameter[0] + " " + parameter[1] + " " + parameter[2] + " " + parameter[3]);
	}

}
