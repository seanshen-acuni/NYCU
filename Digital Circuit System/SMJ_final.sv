module SMJ(
    hand_n0,
    hand_n1,
    hand_n2,
    hand_n3,
    hand_n4,
    out_data
);
input [5:0] hand_n0;
input [5:0] hand_n1;
input [5:0] hand_n2;
input [5:0] hand_n3;
input [5:0] hand_n4;
output logic [1:0] out_data;
wire logic [5:0] a,b,c,d,e,a0,b0,c0,d0,e0,a1,b1,c1,d1,e1,a2,b2,c2,d2,e2,a3,b3,c3,d3,e3,a4,b4,c4,d4,e4;

//initialize all wire
//initial out_data=2'b11;

filter f0(a,hand_n0);
filter f1(b,hand_n1);
filter f2(c,hand_n2);
filter f3(d,hand_n3);
filter f4(e,hand_n4);

//sorting
comparator comp_0_0(  a,  b, a0, b0);
comparator comp_0_1(  c,  d, c0, d0);
assign e0 = e ;
comparator comp_1_0( b0, c0, b1, c1);
comparator comp_1_1( d0, e0, d1, e1);
assign a1 = a0;
comparator comp_2_0( a1, b1, a2, b2);
comparator comp_2_1( c1, d1, c2, d2);
assign e2 = e1;
comparator comp_3_0( b2, c2, b3, c3);
comparator comp_3_1( d2, e2, d3, e3);
assign a3 = a2;
comparator comp_4_0( a3, b3, a4, b4);
comparator comp_4_1( c3, d3, c4, d4);
assign e4 = e3;

always @(*) begin
	if (((a == b) && (a == c) && (a == d) && (a == e)) || ((a == 6'b001111) || (b == 6'b001111) || (c == 6'b001111)
	       	|| (d == 6'b001111) || (e == 6'b001111)))
		out_data = 2'b01;
	
	else if (((a4 == b4) && (a4 == c4) && (d4 == e4)) || ((a4 == b4) && (c4 == d4) && (c4 == e4)))
		out_data = 2'b11;
	
	else if (((b4 == a4 + 6'b000001) && (c4 == a4 + 6'b000010) && (d4 == e4) && (a4[5:4] != 2'b00)) ||
	        ((a4 == b4) && (d4 == c4 + 6'b000001) && (e4 == c4 + 6'b000010) && (c4[5:4] != 2'b00)) ||
		((b4 == a4 + 6'b000001) && (b4 == c4) && (b4 == d4) && (e4 == a4 + 6'b000010)))
		out_data = 2'b10;

	else
		out_data = 2'b00;

end
endmodule

//filter exclude impossible input
module filter(out,in_0);
input [5:0] in_0;
output logic [5:0] out;
//initial out = 6'b000000;
always @(*) begin
        if (in_0 != 6'b000111 && in_0 != 6'b001000 && in_0[3:0] != 4'b1001 && in_0[3:0] != 4'b1010 &&
            in_0[3:0] != 4'b1011 && in_0[3:0] != 4'b1100 && in_0[3:0] != 4'b1101 && in_0[3:0] != 4'b1110 &&
            in_0[3:0] != 4'b1111)
                out = in_0;
        else
                out = 6'b001111;
end
endmodule

//bubble sort
module comparator(in_0,in_1,out_0,out_1);
input        [5:0]  in_0, in_1;
output logic [5:0] out_0,out_1;
assign out_0 = (in_0 <= in_1) ? in_0 : in_1;
assign out_1 = (in_0 <= in_1) ? in_1 : in_0;
endmodule
