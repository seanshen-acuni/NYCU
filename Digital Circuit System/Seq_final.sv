module Seq(
	// Input signals
	clk,
	rst_n,
	in_valid,
	in_data,
	// Output signals
	out_valid,
	out_data
);

//---------------------------------------------------------------------
//   INPUT AND OUTPUT DECLARATION                         
//---------------------------------------------------------------------
input clk, rst_n, in_valid;
input [3:0] in_data;
output logic out_valid;
output logic out_data;

//---------------------------------------------------------------------
//   REG AND WIRE DECLARATION                         
//---------------------------------------------------------------------
logic [3:0]in_data1,in_data2;
//---------------------------------------------------------------------
//   YOUR DESIGN                        
//---------------------------------------------------------------------
always@ (posedge clk or negedge rst_n) begin
	if (!rst_n) begin
		out_data <= 0;
	end else begin
		if (in_valid) begin
			in_data1 <= in_data ;
			in_data2 <= in_data1;
		end else begin
			out_data <= 0;
		end
	end
end
always@ (*) begin
	if ((in_data1 != 'x') || (in_data2 != 'x')) begin
		if ((in_data1 < in_data2) && (in_data2 < in_data)) begin
			out_data  <= 1;
			out_valid <= 1;
		end else begin
			out_data  <= 0;
			out_valid <= 1;
		end else begin 
			out_data  <= 0;
			out_valid <= 0;
		end
end

endmodule
