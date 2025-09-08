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

reg [5:0] temp;

always @* begin
  out_num2 = in_num2;

repeat (5) begin
    if(in_num0 > out_num2) begin
      temp = in_num0;
      in_num0 = out_num2;
      out_num2 = temp;
    end
    if(in_num1 > out_num2) begin
      temp = in_num1;
      in_num1 = out_num2;
      out_num2 = temp;
    end
    if(in_num2 > in_num3) begin
      temp = in_num2;
      in_num2 = in_num3;
      in_num3 = temp;
    end
    if(in_num3 > in_num4) begin
      temp = in_num3;
      in_num3 = in_num4;
      in_num4 = temp;
    end
  end
end
endmodule
