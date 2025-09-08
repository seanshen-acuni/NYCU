module Sort(
	in_num0,
	in_num1,
	in_num2,
	in_num3,
	in_num4,
	out_num
);
input  [5:0] in_num0, in_num1, in_num2, in_num3, in_num4;
output logic [5:0] out_num;

reg [5:0] a, b, c, d, e;

assign a = in_num0;
assign b = in_num1;
assign c = in_num2;
assign d = in_num3;
assign e = in_num4;
assign {a, b} = a <= b ? {a, b} : {b, a};
assign {c, d} = c <= d ? {c, d} : {d, c};
assign {b, d} = b <= d ? {b, d} : {d, b};
assign {a, c} = a <= c ? {a, c} : {c, a};
assign {c, e} = c <= e ? {c, e} : {e, c};
assign {a, b} = a <= b ? {a, b} : {b, a};
assign {d, e} = d <= e ? {d, e} : {e, d};
assign {b, c} = b <= c ? {b, c} : {c, b};
assign {b, d} = b <= d ? {b, d} : {d, b};
assign {c, d} = c <= d ? {c, d} : {d, c};
assign {a, c} = a <= c ? {a, c} : {c, a};
assign {c, e} = c <= e ? {c, e} : {e, c};
assign {b, c} = b <= c ? {b, c} : {c, b};
assign {d, e} = d <= e ? {d, e} : {e, d};
assign {b, c} = b <= c ? {b, c} : {c, b};
assign {b, d} = b <= d ? {b, d} : {d, b};
assign {c, d} = c <= d ? {c, d} : {d, c};
assign {a, b} = a <= b ? {a, b} : {b, a};
assign {c, e} = c <= e ? {c, e} : {e, c};
assign {a, c} = a <= c ? {a, c} : {c, a};
assign {b, c} = b <= c ? {b, c} : {c, b};
assign {d, e} = d <= e ? {d, e} : {e, d};
assign {b, c} = b <= c ? {b, c} : {c, b};
assign {b, d} = b <= d ? {b, d} : {d, b};
assign {c, d} = c <= d ? {c, d} : {d, c};
assign {a, b} = a <= b ? {a, b} : {b, a};
assign {c, e} = c <= e ? {c, e} : {e, c};
assign {b, c} = b <= c ? {b, c} : {c, b};
assign {d, e} = d <= e ? {d, e} : {e, d};
assign {b, c} = b <= c ? {b, c} : {c, b};

assign out_num = c;
endmodule
