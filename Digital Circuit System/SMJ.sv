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

logic [5:0] a,b,c,d,e,a1,b1,c1,d1,e1;
filter f1(a,hand_n0);
filter f2(b,hand_n1);
filter f3(c,hand_n2);
filter f4(d,hand_n3);
filter f5(e,hand_n4);

always @(*) begin
	if(((a == b) && (a == c) && (a == d) && (a == e)) || (a == 6'b001111) || (b == 6'b001111) || (c == 6'b001111)
	       	|| (d == 6'b001111) || (e == 6'b001111) )
		out_data = 2'b01;
	else
		out_data = 2'b00;

end
endmodule
//filter exclude impossible input
module filter(out,in_0);
input [5:0] in_0;
output logic [5:0] out;

always @(*) begin
        if (in_0!=6'b000111 && in_0!=6'b001000 && in_0!=6'b??1001 && in_0!=6'b??1010
        && in_0!=6'b??1011 && in_0!=6'b??1100 && in_0!=6'b??1101 && in_0!=6'b??1110
        && in_0!=6'b??1111)
                out = in_0;
        else
                out = 6'b001111;
end
endmodule

