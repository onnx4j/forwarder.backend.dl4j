package org.forwarder.backend.impls.dl4j.opsets;

import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.TreeMap;

import org.forwarder.backend.impls.dl4j.opsets.OpsetTest.MarkdownTable.BodyRow;
import org.forwarder.backend.impls.dl4j.opsets.OpsetTest.MarkdownTable.HeaderItem;
import org.forwarder.backend.impls.dl4j.opsets.OpsetTest.MarkdownTable.HeaderItem.ALigned;
import org.forwarder.backend.impls.dl4j.opsets.OpsetTest.MarkdownTable.RowItem;
import org.junit.Test;
import org.onnx4j.opsets.domain.aiOnnx.AiOnnxOperator;
import org.reflections.Reflections;

public class OpsetTest {

	@Test
	public void buildOpsetTbl() throws Throwable {
		Map<String, List<Integer>> supportedOps = new LinkedHashMap<String, List<Integer>>();

		Reflections reflections = new Reflections(
				"org.forwarder.backend.impls.dl4j");
		Set<Class<? extends AiOnnxOperator>> opClasses = reflections
				.getSubTypesOf(AiOnnxOperator.class);

		long maxSupportedVer = 0L;
		for (Class<? extends AiOnnxOperator> opClz : opClasses) {
			if (opClz.isInterface())
				continue;

			AiOnnxOperator operator = createAiOnnxOperator(opClz);
			long ver = operator.getVersion();
			if (ver > maxSupportedVer)
				maxSupportedVer = ver;
		}
		for (Class<? extends AiOnnxOperator> opClz : opClasses) {
			if (opClz.isInterface())
				continue;

			AiOnnxOperator operator = createAiOnnxOperator(opClz);
			String opType = operator.getOpType();
			long ver = operator.getVersion();

			List<Integer> supportedList = supportedOps.get(opType);
			if (supportedList == null) {
				supportedList = new LinkedList<Integer>();
				supportedOps.put(opType, supportedList);
				for (int n = 0; n < maxSupportedVer; n++) {
					supportedList.add(0);
				}
			}
			supportedList.set((int) ver - 1, (int) ver);
		}
		Map<String, List<Integer>> sortedOps = new TreeMap<String, List<Integer>>(
				supportedOps);

		HeaderItem[] headers = new HeaderItem[(int) (maxSupportedVer + 1)];
		headers[0] = new HeaderItem("Operator", ALigned.LEFT);
		for (int n = 1; n < headers.length; n++) {
			headers[n] = new HeaderItem("Opset" + n, ALigned.CENTER);
		}

		MarkdownTable tbl = new MarkdownTable(headers);
		for (Entry<String, List<Integer>> entrySet : sortedOps.entrySet()) {
			BodyRow row = new BodyRow();
			row.add(new RowItem(entrySet.getKey()));

			long currSupportVer = 0l;
			for (Integer supportVer : entrySet.getValue()) {
				String data = "";
				if (supportVer > 0)
					currSupportVer = supportVer;

				if (currSupportVer > 0)
					data = String.valueOf(currSupportVer);
				else
					data = "-";
				row.add(new RowItem(data));
			}
			tbl.addRow(row);
		}
		System.out.println(tbl.toString());
	}

	private AiOnnxOperator createAiOnnxOperator(
			Class<? extends AiOnnxOperator> opClz)
			throws InstantiationException, IllegalAccessException {
		return opClz.newInstance();
	}

	static class MarkdownTable {

		static class HeaderItem {

			enum ALigned {
				LEFT, CENTER, RIGHT
			}

			private String name;
			private ALigned aligned;

			public HeaderItem(String name) {
				this(name, ALigned.LEFT);
			}

			public HeaderItem(String name, ALigned aligned) {
				super();
				this.name = name;
				this.aligned = aligned;
			}

			public String getName() {
				return name;
			}

			public void setName(String name) {
				this.name = name;
			}

			public ALigned getAligned() {
				return aligned;
			}

			public void setAligned(ALigned aligned) {
				this.aligned = aligned;
			}
		}

		static class BodyRow {
			private List<RowItem> items = new LinkedList<OpsetTest.MarkdownTable.RowItem>();

			public BodyRow add(RowItem item) {
				this.items.add(item);
				return this;
			}

			public List<RowItem> getItems() {
				return items;
			}
		}

		static class RowItem {
			private String data;

			public RowItem(String data) {
				super();
				this.data = data;
			}

			public String getData() {
				return data;
			}

			public void setData(String data) {
				this.data = data;
			}
		}

		private List<HeaderItem> header = new LinkedList<OpsetTest.MarkdownTable.HeaderItem>();
		private List<BodyRow> body = new LinkedList<OpsetTest.MarkdownTable.BodyRow>();

		public MarkdownTable(HeaderItem... headers) {
			for (HeaderItem header : headers) {
				this.header.add(header);
			}
		}

		public MarkdownTable(String... headerNames) {
			for (String headerName : headerNames) {
				this.header.add(new HeaderItem(headerName));
			}
		}

		public MarkdownTable addRow(BodyRow row) {
			this.body.add(row);
			return this;
		}

		@Override
		public String toString() {
			StringBuffer stringBuf = new StringBuffer();

			stringBuf.append("|");
			for (HeaderItem header : this.header) {
				stringBuf.append(header.getName());
				stringBuf.append("|");
			}

			stringBuf.append("\n|");
			for (HeaderItem header : this.header) {
				switch (header.getAligned()) {
				default:
				case LEFT:
					stringBuf.append(":---");
					break;
				case CENTER:
					stringBuf.append(":---:");
					break;
				case RIGHT:
					stringBuf.append("---:");
					break;
				}
				stringBuf.append("|");
			}

			for (BodyRow row : this.body) {
				stringBuf.append("\n|");
				for (RowItem item : row.getItems()) {
					stringBuf.append(item.getData());
					stringBuf.append("|");
				}
			}

			return stringBuf.toString();
		}
	}

}
